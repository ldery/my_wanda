import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import pdb
from time import time
import datetime
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
# from lib.modelling_llama import LlamaForCausalLM
from lib.modelling_llama_mod import LlamaForCausalLM
# from lib.my_prune import my_check_sparsity, my_method_prune
from lib.eval import eval_ppl, eval_ppl_trainonly
from collections import defaultdict
import pickle as pkl
import random
from lib.scoring_model import ScoreModel
import wandb

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

def hook_fn(module_name, info_cache):
	def hook(module, in_, out_):
		if isinstance(in_, tuple):
			in_ = in_[0]

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

def get_score_models(score_perfs, module_map, info_cache, hp_dict, wandb_run, parent_id='.'):
	score_map = {}
	for id_, (name, module) in module_map.items():
		# Get a score map
		print('Working on layer ', id_)
		base_mask = info_cache[name]['in'][1] / info_cache[name]['in'][0]
		base_mask = (base_mask.squeeze().float() * module.main_mask.squeeze().float()).view(-1, 1)
		base_mask = base_mask / base_mask.sum()
		sm = ScoreModel(id_='{}/{}'.format(parent_id, id_), num_players=module.intermediate_size, base_mask=base_mask, hp_dict=hp_dict, wandb=wandb_run)
		sm.cuda()
		run_info = score_perfs[0][id_], score_perfs[1][id_]
		sm.update_with_info(run_info)
		score_map[id_] = (sm.base_model.score_tensor.clone()).squeeze().detach()
	return score_map


def investigate_rms_fix(args, model, wandb_run):

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
	info_cache, hook_handles = defaultdict(dict), []
	for (name, module) in model.named_modules():
		# For now, only focus on the MLPs
		if name.endswith('mlp'):
			hook_handles.append(module.register_forward_hook(hook_fn(name, info_cache)))

	module_map = {}
	for (name, module) in model.named_modules():
		# For now, only focus on the MLPs
		if name.endswith('mlp'):
			id_  = int(name.split('.')[2])
			module_map[id_] = (name, module)
			intermediate_sz = module.intermediate_size

	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
	eval_ppl_trainonly(model, tokenizer, bsz=args.bsz, nsamples=args.nsamples)
	
	hp_dict = get_linearmodel_hpdict(args)
	score_matrix = defaultdict(lambda: None)
	all_sampling_proba = defaultdict(lambda: np.ones((intermediate_sz)))
	compute_updated_masks_local(args.prune_frac, score_matrix, all_sampling_proba, None)

	n_iter = int(np.floor(np.log(args.sparsity_ratio)/ np.log(1 - args.prune_frac))) - 1
	for i in range(n_iter):
		start = time()
		score_info = get_random_mask_scores(
							model, tokenizer, module_map, all_sampling_proba,
							bsz=args.bsz, nsamples=args.nsamples,
							mpi=args.masks_per_iter, pfrac=args.prune_frac
		)
		score_model_maps = get_score_models(score_info, module_map, info_cache, hp_dict, wandb_run, parent_id='Iter.{}'.format(i))
		gen_scores_time = time() - start
		print('It took ', str(datetime.timedelta(seconds=gen_scores_time)), ' to complete score generation and linear model fit')
		start = time()
		# Need to do some fitting to a linear model here.
		compute_updated_masks_local(args.prune_frac, score_matrix, all_sampling_proba, score_model_maps)
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=args.bsz, nsamples=args.nsamples)
		time_delta = time() - start
		
		wandb_run.log({'SysStats/scoreruntime': gen_scores_time, 'SysStats/pruneruntime': time_delta, 'Pruning/trainPPL': this_ppl})
		print('It took ', str(datetime.timedelta(seconds=time_delta)), ' to complete mask updated and eval of updated model')
		print('[Iter = {}] Took : {} | Achieved train ppl: '.format(i, str(datetime.timedelta(seconds=(time_delta + gen_scores_time)))), this_ppl, flush=True)
		print('===='*20)
	
	ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device)
	wandb_run.log({'Final/TrainPPL': ppl_train, 'Final/TestPPL': ppl_test})
	print('[Wikitext][After] Train PPL = {:.3f} | Test PPL = {:.3f}'.format(ppl_train, ppl_test))

	for handle in hook_handles:
		handle.remove()

	# Now save all the required information
	mask_info = {name: module.main_mask for _, (name, module) in module_map.items()}
	save_loc = os.path.join(args.save, 'mask_info.pkl')
	with open(save_loc, 'wb') as handle:
		pkl.dump(mask_info, handle)

def args_to_dict(args):
	return {
		'nsamp': args.nsamples,
		'sparsity': args.sparsity_ratio,
		'prune_frac': args.prune_frac,
		'bsz': args.bsz,
		'masksperiter': args.masks_per_iter,
		'LinModel.regweight': args.sm_reg_weight,
		'LinModel.lr': args.sm_lr,
		'LinModel.bsz': args.sm_bsz,
		
	}

def args_to_str(args):
	relevant_args = args_to_dict(args)
	return '_'.join(['{}={}'.format(k, v) for k, v in relevant_args.items()])

def get_linearmodel_hpdict(args):
	base_hp = {
		'lr' :args.sm_lr,
		'reg_weight': args.sm_reg_weight,
		'bsz' : args.sm_bsz,
		'nepochs' : args.sm_nepochs,
		'patience': 10,
	}
	return base_hp

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf', help='LLaMA model')
	parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
	parser.add_argument('--nsamples', type=int, default=14, help='Number of calibration samples.')
	parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
	parser.add_argument('--prune_frac', type=float, default=0.1, help='Fraction of weights to prune at a time')
	parser.add_argument('--bsz', type=int, default=14, help='Instantaneous batch size for forward pass')

	parser.add_argument("--prune_method", type=str, choices=["ours", "magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])
	parser.add_argument("--cache_dir", default="llm_weights", type=str )
	parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
	parser.add_argument('--save', type=str, default=None, help='Path to save results.')
	parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
	parser.add_argument('--masks_per_iter', type=int, default=10, help='How many masks to generate per-iteration')
	
	# Hyperparams for scoring model
	parser.add_argument('--sm_reg_weight', type=float, default=0.0, help='reg-weight to use')
	parser.add_argument('--sm_lr', type=float, default=5e-4, help='lr to use for fitting linear model')
	parser.add_argument('--sm_bsz', type=int, default=256, help='batch size for fitting linear model')
	parser.add_argument('--sm_nepochs', type=int, default=50, help='number of epochs to use to fit the linear model')
	
	# Wandb HP
	parser.add_argument('--wandb_project_name', type=str, default='Prune-No-Backward', help='Wandb project name')

	args = parser.parse_args()
	print(args)
	str_of_args = args_to_str(args)
	args.save = os.path.join(args.save, str_of_args)
	os.makedirs(args.save, exist_ok=True)


	# Setting seeds for reproducibility
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)

	wandb_run = wandb.init(
		project=args.wandb_project_name,
		name=str_of_args,
		config=args_to_dict(args),
	)

	model_name = args.model.split("/")[-1]
	print(f"loading llm model {args.model}")
	model = get_llm(args.model, args.cache_dir)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
	investigate_rms_fix(args, model, wandb_run)
	print('Done and exitting')


if __name__ == '__main__':
    main()


