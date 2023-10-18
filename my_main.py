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
from lib.eval import eval_ppl, eval_ppl_trainonly
from collections import defaultdict
import pickle as pkl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())
# def compute_updated_masks_global(prune_frac=0.5):
# 	aggregated = []
# 	for (name, module) in model.named_modules():
# 		if name.endswith('mlp'): # TODO -- ldery -- make some changes here
# 			avg_act_magnitudes = info_cache[name]['in'][1] / info_cache[name]['in'][0]
# 			if module.layer_mask is not None:
# 				avg_act_magnitudes = (module.layer_mask * avg_act_magnitudes)
# 			aggregated.append(avg_act_magnitudes[avg_act_magnitudes > 0])

# 	joint_ = torch.concat(aggregated).float()
# 	qt = torch.quantile(joint_, prune_frac)
# 	for (name, module) in model.named_modules():
# 		if name.endswith('mlp'): # TODO -- ldery -- make some changes here
# 			avg_act_magnitudes = info_cache[name]['in'][1] / info_cache[name]['in'][0]
# 			mask_ = ((avg_act_magnitudes > qt)*1.0).half()
# # 				if '31' in name:
# 			print('[{}] - updated sparsity : '.format(name), mask_.mean())
# 			module.layer_mask = mask_

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

def investigate_rms_fix(model_name, model, bsz=6, nsamples=128):
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
	
	def update_mask_one_layer(module, info, prune_frac):
		avg_act_magnitudes = info['in'][1] / info['in'][0]
		if module.layer_mask is not None:
			avg_act_magnitudes = (module.layer_mask * avg_act_magnitudes)

		non_zero_mag = avg_act_magnitudes[avg_act_magnitudes > 0]
		qt = torch.quantile(non_zero_mag.squeeze().float(), prune_frac)
		mask_ = ((avg_act_magnitudes > qt)*1.0).half()
		module.layer_mask = mask_
		return mask_.mean().item()

	def compute_updated_masks_local(prune_frac):
		for id_, (name, module) in module_map.items():
			update_mask_one_layer(module, info_cache[name], prune_frac)

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

	prune_frac = 0.2
	target = 0.5
	n_iter = int(np.ceil(np.log(target)/ np.log(1 - prune_frac)))
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
	n_layers = 32
	for i in range(n_iter):
		for j in range(32): #(n_layers - 1, -1, -1):
			this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=32)
			m_name, module = module_map[j]
			occupancy = update_mask_one_layer(module, info_cache[m_name], prune_frac)
			print('Layer {} - Occ - {:.3f}. Orig PPL - {:.3f}'.format(j, occupancy, this_ppl))
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples)
# 		compute_updated_masks_local(prune_frac)
		print('[{}] Achieved train ppl: '.format(i), this_ppl)
		info_cache = defaultdict(dict)
# 		compute_updated_masks_local(prune_frac)

	for handle in hook_handles:
		handle.remove()
# 	with open('og_llama_nsamples={}.pkl'.format(nsamples), 'wb') as handle:
# 		pkl.dump(info_cache, handle)

def investigate_model(model_name, model, bsz=12, nsamples=512):
	info_cache = defaultdict(dict)
	def hook_fn(module_name):
		def hook(module, in_, out_):
			if isinstance(in_, tuple):
				in_ = in_[0]

			if not isinstance(in_, torch.Tensor):
				pdb.set_trace()

			flat_in = in_.view(-1, in_.shape[-1]).mean(axis=0).abs()
			flat_out = out_.view(-1, out_.shape[-1]).mean(axis=0).abs()
			if 'in' not in info_cache[module_name]:
				info_cache[module_name]['in'] = [1, flat_in]
				info_cache[module_name]['out'] = [1, flat_out]
			else:
				info_cache[module_name]['in'] = [
					info_cache[module_name]['in'][0] + 1,  
					info_cache[module_name]['in'][1].add_(flat_in)
				]
				info_cache[module_name]['out']= [
					info_cache[module_name]['out'][0] + 1,
					info_cache[module_name]['out'][1].add_(flat_in)
				]
		return hook

	# add forward hooks
	remove_handles = []
	for (name, module) in model.named_modules():
		if name.endswith('mlp'):
			remove_handles.append(module.register_forward_hook(hook_fn(name)))

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
	eval_ppl_trainonly(model, tokenizer, bsz=bsz, nsamples=nsamples)
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
	return info_cache


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
	args = parser.parse_args()

	# Setting seeds for reproducibility
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)

	model_name = args.model.split("/")[-1]
	print(f"loading llm model {args.model}")
	model = get_llm(args.model, args.cache_dir)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
	investigate_rms_fix(args.model, model)
	print('Done and exitting')
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