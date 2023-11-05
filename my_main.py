import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import pdb
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
# from lib.modelling_llama import LlamaForCausalLM
from lib.modelling_llama_mod import LlamaForCausalLM
# from lib.my_prune import my_check_sparsity, my_method_prune
from lib.eval import eval_ppl, eval_ppl_trainonly, train_wikitext
import gc

from collections import defaultdict
import pickle as pkl
import random
from transformers.pytorch_utils import  prune_linear_layer
from lib.scoring_model import ScoreModel
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def set_masks(module_map, all_masks, all_sampling_proba, pfrac=0.1):
	for k, (name, module) in module_map.items():
		module.is_using_main = False
		sampling_proba = all_sampling_proba[k]
		mask = get_random_mask(module.intermediate_size, module.main_mask, sampling_proba, pfrac)
		all_masks[k].append(torch.Tensor(mask).squeeze())
		module.temp_mask = torch.Tensor(mask).type(module.up_proj.weight.type())

def get_random_mask(intermediate_sz, main_mask, sampling_proba, pfrac):
	init_set = np.ones((1, 1, intermediate_sz)) if main_mask is None else main_mask.cpu().numpy()
	num_to_zero = int(pfrac * np.sum(init_set)) + 1
	non_zero_idxs = np.squeeze(init_set).nonzero()[0]
	new_proba = sampling_proba[non_zero_idxs]
	new_proba = new_proba / np.sum(new_proba)
	chosen_idxs = np.random.choice(non_zero_idxs, size=num_to_zero, p=new_proba)
	init_set[:, :, chosen_idxs] = 0
	return init_set

def get_random_mask_scores(model, tokenizer, module_map, all_sampling_proba, bsz=12, nsamples=32, mpi=100, pfrac=0.1):

	# set to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = False

	all_masks, all_perfs = defaultdict(list), defaultdict(list)
	seed_ = random.randint(0, 1e4)
	for iter_ in range(mpi):
		# set the layer mask here
		set_masks(module_map, all_masks, all_sampling_proba, pfrac=pfrac)
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples, seed=seed_)
		print('Iter : ', iter_, ' PPL = ', this_ppl)
		for k, (name, module) in module_map.items():
			# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
			all_perfs[k].append(-this_ppl)

	# reset to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = True

	return all_masks, all_perfs

def get_llm(model_name, cache_dir="llm_weights"):
	model = LlamaForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.float16, 
		cache_dir=cache_dir, 
		low_cpu_mem_usage=True, 
		device_map="auto"
	)

	model.seqlen = model.config.max_position_embeddings 
	return model

def get_score_models(score_perfs, module_map, info_cache, reg_weight=1e-3):
	score_map = {}
	for id_, (name, module) in module_map.items():
		# Get a score map
		print('Working on layer ', id_)
		base_mask = info_cache[name]['in'][1] / info_cache[name]['in'][0]
		base_mask = (base_mask.squeeze().float() * module.main_mask.squeeze().float()).view(-1, 1)
		base_mask = base_mask / base_mask.sum()
		sm = ScoreModel(num_players=module.intermediate_size, base_mask=base_mask, reg_weight=reg_weight)
		sm.cuda()
		run_info = score_perfs[0][id_], score_perfs[1][id_]
		sm.update_with_info(run_info)
		score_map[id_] = (sm.base_model.score_tensor.clone()).squeeze().detach()

	return score_map

def investigate_rms_fix(model_name, model, bsz=14, nsamples=14, masks_per_iter=20, reg_weight=1e-1):

	info_cache = defaultdict(dict)

	# TODO [ldery] -- double check effect of not aggregating info_cache
	def hook_fn(module_name):
		def hook(module, in_, out_):
			if isinstance(in_, tuple):
				in_ = in_[0]

			if not isinstance(in_, torch.Tensor):
				pdb.set_trace()

			flat_in = module.intermed_cache
			module.intermed_cache = None
			if 'in' not in info_cache[module_name]:
				info_cache[module_name]['in'] = [1, flat_in]
			else:
				info_cache[module_name]['in'] = [
					info_cache[module_name]['in'][0] + 1,  
					info_cache[module_name]['in'][1].add_(flat_in)
				]
		return hook
	
	def update_mask_one_layer(module, info, score_info, prune_frac, score_model_weights):
		if score_model_weights is None:
			score_model_weights = (info['in'][1] / info['in'][0]).squeeze()

		if module.main_mask is not None:
			qt = torch.quantile((score_model_weights[(module.main_mask).squeeze() > 0]).squeeze().float(), prune_frac)
		else:
			qt = torch.quantile(score_model_weights.squeeze().float(), prune_frac)

		mask_ = ((score_model_weights > qt)*1.0).half()
		if module.main_mask is not None:
			module.main_mask *= (mask_).view(info['in'][1].shape)
		else:
			module.main_mask = (mask_).view(info['in'][1].shape)

		avg_act_magnitudes = info['in'][1] / info['in'][0]
		sampling_proba = avg_act_magnitudes.cpu().float().squeeze().numpy()
		sampling_proba = sampling_proba.max() - sampling_proba
		sampling_proba *= (module.main_mask).cpu().float().squeeze().numpy()
		sampling_proba /= np.sum(sampling_proba)
		if np.isnan(sampling_proba).any():
			pdb.set_trace()
		return module.main_mask.mean().item(), sampling_proba

	def compute_updated_masks_local(prune_frac, score_matrix, all_sampling_proba, score_model_maps):
		avgs = 0.0
		for id_, (name, module) in module_map.items():
			score_model = None if score_model_maps is None else score_model_maps[id_]
			(this_avg, new_samp_prob) = update_mask_one_layer(
											module, info_cache[name], 
											score_matrix[id_],  prune_frac,
											score_model)
			avgs += this_avg
			all_sampling_proba[id_] = new_samp_prob
		print('The new occupacy is : {:.3f}'.format(avgs / len(module_map)))

		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			info_cache[k] = dict()


	# add forward hooks
	hook_handles = []
	for (name, module) in model.named_modules():
		if name.endswith('mlp'):
			hook_handles.append(module.register_forward_hook(hook_fn(name)))

	module_map = {}
	for (name, module) in model.named_modules():
		if name.endswith('mlp'):
			id_  = int(name.split('.')[2])
			module_map[id_] = (name, module)
			intermediate_sz = module.intermediate_size

	print('We are using this reg weight : ', reg_weight)
	prune_frac = 0.95
	target = 0.05
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
	eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples)
	score_matrix = defaultdict(lambda: None)
	all_sampling_proba = defaultdict(lambda: np.ones((intermediate_sz)))
	score_model_maps = None
	compute_updated_masks_local(prune_frac, score_matrix, all_sampling_proba, score_model_maps)

	n_iter = int(np.floor(np.log(target)/ np.log(1 - prune_frac))) - 1
	for i in range(n_iter):
		score_info = get_random_mask_scores(
							model, tokenizer, module_map, all_sampling_proba,
							bsz=bsz, nsamples=nsamples,
							mpi=masks_per_iter, pfrac=prune_frac
		)
# 		score_model_maps = get_score_models(score_info, module_map, info_cache, reg_weight=reg_weight)
		# Need to do some fitting to a linear model here.
		compute_updated_masks_local(prune_frac, score_matrix, all_sampling_proba, score_model_maps)
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples)
		print('[{}] Achieved train ppl: '.format(i), this_ppl)

# 	ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device)
# 	print('[Wikitext][After] Train PPL = {:.3f} | Test PPL = {:.3f}'.format(ppl_train, ppl_test))

	# Print out the parameters
	num_params = sum([p.numel() for p in model.parameters()])

	# We want to perform pruning here !
	for m_id, (name, module) in module_map.items():
		index = module.main_mask.squeeze().nonzero().squeeze()
		new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
		module.gate_proj = None
		module.gate_proj = new_gate_proj
		new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
		module.up_proj  = None
		module.up_proj = new_up_proj
		new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
		module.down_proj = None
		module.down_proj = new_down_proj
		module.pre_prune = False
		module.main_mask = None
		module.temp_mask = None
		module.intermed_cache = None

	for hook in hook_handles:
		hook.remove()
	torch.cuda.empty_cache()
	updated_numparams = sum([p.numel() for p in model.parameters()])
	print('Og Params = {:.3f}'.format(num_params / 1e9), 'Updated Params = {:.3f}'.format(updated_numparams / 1e9))
	
	ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device)
	print('[Wikitext][Post-Pruning] Train PPL = {:.3f} | Test PPL = {:.3f}'.format(ppl_train, ppl_test))

	for handle in hook_handles:
		handle.remove()

	info_cache.clear()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='LLaMA model')
	parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
	parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
	parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')

	parser.add_argument("--prune_method", type=str, choices=["ours", "magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])
	parser.add_argument("--cache_dir", default="llm_weights", type=str )
	parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
	parser.add_argument('--save', type=str, default=None, help='Path to save results.')
	parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
	parser.add_argument('--masks_per_iter', type=int, default=10, help='How many masks to generate per-iteration')
	parser.add_argument('--reg_weight', type=float, default=0.0, help='reg-weight to use')
	parser.add_argument('--ft_lr', type=float, default=5e-5, help='finetuning learning rate')
	args = parser.parse_args()

	# Setting seeds for reproducibility
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)

	model_name = args.model.split("/")[-1]
	print(f"loading llm model {args.model}")
	model = get_llm(args.model, args.cache_dir)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
	investigate_rms_fix(args.model, model, nsamples=args.nsamples, masks_per_iter=args.masks_per_iter, reg_weight=args.reg_weight)
	print('Done Prunining -- Now FineTuning')
	gc.collect()
	torch.cuda.empty_cache()
	model.train()
	_, final_test_ppl = train_wikitext(model, tokenizer, device=model.device, lr_=args.ft_lr)
	print('[Wikitext][Post-FT] Test PPL = {:.3f}'.format(final_test_ppl))
	
	
# 	investigate_model(args.model, model)
# 	print('Done and exitting')
# 	device = torch.device("cuda:0")
# 	if "30b" in args.model or "65b" in args.model:
# 		# for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
# 		device = model.hf_device_map["lm_head"]
# 	print("use device ", device)

# # 	if args.sparsity_ratio != 0:
# # 		print("pruning starts")
# # 		my_method_prune_(args, model, tokenizer, device)

# # 	################################################################
# # 	print("*"*30)
# # 	sparsity_ratio = my_check_sparsity(model)
# # 	print(f"sparsity sanity check {sparsity_ratio:.4f}")
# # 	print("*"*30)
# # 	################################################################
# 	ppl_train, ppl_test = eval_ppl(model, tokenizer, device)
# 	print(f"ppl on wikitext_train {ppl_train}, wikitext_test {ppl_test}")

# # 	if not os.path.exists(args.save):
# # 		os.makedirs(args.save)
# # 	save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
# # 	with open(save_filepath, "w") as f:
# # 		if "ablate" in args.prune_method:
# # 			print("method\tactual_sparsity\tppl_train\tppl_test", file=f, flush=True)
# # 			print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_train:.4f}\t{ppl_test:.4f}", file=f, flush=True)
# # 		else:
# # 			print("method\tactual_sparsity\tppl_test", file=f, flush=True)
# # 			print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

# # 	if args.save_model:
# # 		model.save_pretrained(args.save_model)
# # 		tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()


# def investigate_model(model_name, model, bsz=12, nsamples=512):
# 	info_cache = defaultdict(dict)
# 	def hook_fn(module_name):
# 		def hook(module, in_, out_):
# 			if isinstance(in_, tuple):
# 				in_ = in_[0]

# 			if not isinstance(in_, torch.Tensor):
# 				pdb.set_trace()

# 			flat_in = in_.view(-1, in_.shape[-1]).mean(axis=0).abs()
# 			flat_out = out_.view(-1, out_.shape[-1]).mean(axis=0).abs()
# 			if 'in' not in info_cache[module_name]:
# 				info_cache[module_name]['in'] = [1, flat_in]
# 				info_cache[module_name]['out'] = [1, flat_out]
# 			else:
# 				info_cache[module_name]['in'] = [
# 					info_cache[module_name]['in'][0] + 1,  
# 					info_cache[module_name]['in'][1].add_(flat_in)
# 				]
# 				info_cache[module_name]['out']= [
# 					info_cache[module_name]['out'][0] + 1,
# 					info_cache[module_name]['out'][1].add_(flat_in)
# 				]
# 		return hook

# 	# add forward hooks
# 	remove_handles = []
# 	for (name, module) in model.named_modules():
# 		if name.endswith('mlp'):
# 			remove_handles.append(module.register_forward_hook(hook_fn(name)))

# 	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# 	eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples)
# 	for handle in remove_handles:
# 		handle.remove()

# 	with open('og_llama_nsamples={}.pkl'.format(nsamples), 'wb') as handle:
# 		pkl.dump(info_cache, handle)
# 	updated_cache = 
# 	for k, v in info_cache.items():
# 		temp_out =  (v['out'][1] / v['out'][0]).cpu().numpy()
# 		print(k, temp_out.min(), temp_out.max(), np.percentile(temp_out, [25,75]))

# 	pdb.set_trace()
# 	print('This is a short test')
# 	return info_cache
