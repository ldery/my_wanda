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
from lib.latest_modelling_llama import LlamaForCausalLM
# from lib.modelling_llama_mod import LlamaForCausalLM
from lib.data import get_loaders
# from lib.my_prune import my_check_sparsity, my_method_prune
from lib.eval import eval_ppl, eval_ppl_trainonly, eval_ppl_train
from collections import defaultdict
import pickle as pkl
import random
from lib.scoring_model import ScoreModelHP
import wandb
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import gc
import random
from pprint import pprint

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

INF = 1e8

def set_masks(module_map, all_masks, module_index_info, global_proba, pfrac=0.1, mlp_attn_ratio=1.0, use_complement=False):
	# Sample from the global distribution here!
	init_set = np.ones_like(global_proba)
	num_to_zero = int(pfrac * np.sum(init_set)) + 1
	chosen_idxs = np.random.choice(np.arange(len(global_proba)), size=num_to_zero, p=global_proba, replace=False)
	init_set[chosen_idxs] = 0

	init_set_start = 0
	for k, (name, module) in module_map.items():
		this_pfrac = pfrac
		if name.endswith('self_attn'):
			this_pfrac = pfrac * mlp_attn_ratio

		module.is_using_main = False
		_, fixed_indices, use_indices = module_index_info[k]
		if use_complement:
			module.temp_mask = 1 - module.temp_mask
			module.temp_mask[:, :, fixed_indices] = 1.0
			all_masks[k].append(module.temp_mask.cpu().squeeze()[use_indices])
		else:
			init_set_end = init_set_start + len(use_indices)
			sampled_mask = init_set[init_set_start: init_set_end]
			init_set_start = init_set_end
			mask = torch.ones_like(module.main_mask)
			mask[:, :, use_indices] = torch.tensor(sampled_mask).type(module.main_mask.type()).view(*(mask[:, :, use_indices]).shape)
			module.temp_mask = torch.Tensor(mask).type(module.main_mask.type())
			all_masks[k].append(torch.Tensor(sampled_mask))

	if not use_complement:
		assert init_set_end == len(init_set)

def get_train_ppl_multitry(model, trainloader, this_bsz):
	continue_ = True
	while continue_:
		with torch.no_grad():
			try:
				this_ppl = eval_ppl_train(model, trainloader, bs=this_bsz, device=torch.device("cuda:0"))
				continue_ = False
			except Exception as e:
				if 'memory' in str(e):
					print("Encountered a memory issue. Scaling bsz from {} to {}".format(this_bsz, max(1, this_bsz // 2)))
					gc.collect()
					torch.cuda.empty_cache()
					this_bsz = max(1, this_bsz // 2)
				else:
					print(e)
					exit()

	return this_ppl, this_bsz

def get_random_mask_scores(model, dataset, module_map, module_index_info, global_proba, use_complement=True, bsz=12, nsamples=32, mpi=100, pfrac=0.1, mlp_attn_ratio=1.0):

	# set to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = False

	all_masks, all_perfs = defaultdict(list), defaultdict(list)
	seed_ = random.randint(0, 1e4)
	niters = mpi // 2 if use_complement else mpi
	for iter_ in range(niters):
		this_bsz = bsz

		# set the layer mask here
		set_masks(module_map, all_masks, module_index_info, global_proba, pfrac=pfrac, mlp_attn_ratio=mlp_attn_ratio)
		this_ppl, this_bsz = get_train_ppl_multitry(model, dataset, this_bsz)

		print('[v1]Iter : ', iter_, ' PPL = ', this_ppl)
		this_ppl = this_ppl if this_ppl < INF else INF

		for k, (name, module) in module_map.items():
			# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
			all_perfs[k].append(-this_ppl)

		if use_complement:
			# set the complement mask here
			set_masks(
				module_map, all_masks, module_index_info, global_proba, pfrac=pfrac,
				mlp_attn_ratio=mlp_attn_ratio, use_complement=True
			)
			this_ppl, this_bsz = get_train_ppl_multitry(model, dataset, this_bsz)

			print('[v2]Iter : ', iter_, ' PPL = ', this_ppl)
			this_ppl = this_ppl if this_ppl < INF else INF

			for k, (name, module) in module_map.items():
				# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
				all_perfs[k].append(-this_ppl)

	# reset to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = True

	return all_masks, all_perfs


def get_llm(model_name, cache_dir="llm_weights", prune_seqlen=1024):

	model = LlamaForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch.float16,
		cache_dir=cache_dir,
		low_cpu_mem_usage=True,
		device_map="auto"
	)

	model.seqlen = prune_seqlen
	print('xx'*20)
	print('This is the current model sequence length: ', model.seqlen)
	print('xx'*20)
	return model

def hook_fn(module_name, info_cache):
	def hook(module, in_, out_):
		if isinstance(module.intermed_cache, tuple):
			flat_in = (module.intermed_cache[0]).clone().detach().float(), (module.intermed_cache[1]).clone().detach().float()
			if 'in' not in info_cache[module_name]:
				info_cache[module_name]['in'] = [1, flat_in]
			else:
				info_cache[module_name]['in'] = [
					info_cache[module_name]['in'][0] + 1,
					(info_cache[module_name]['in'][1][0].add_(flat_in[0]), info_cache[module_name]['in'][1][1].add_(flat_in[1]))
				]
		else:
			flat_in = (module.intermed_cache).clone().detach().float()
			if 'in' not in info_cache[module_name]:
				info_cache[module_name]['in'] = [1, flat_in]
			else:
				info_cache[module_name]['in'] = [
					info_cache[module_name]['in'][0] + 1,
					info_cache[module_name]['in'][1].add_(flat_in)
				]
		module.intermed_cache = None
	return hook

def get_score_models(score_perfs, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='.',  model_type='local'):
	score_map = {}
	# do some global modelling here
	# aggregate all the data here
	xs = None
	for id_, (name, module) in module_map.items():
		if xs is None:
			xs = score_perfs[0][id_]
		else:
			xs = [torch.cat((xs[k], score_perfs[0][id_][k])) for k in range(len(xs))]
	xs = [k.cuda() for k in xs]
	ys = score_perfs[1][id_]

	is_valid = np.array(ys) < INF
	this_masks, this_scores = [], []
	for idx, truth_val in enumerate(is_valid):
		if truth_val:
			this_masks.append(xs[idx])
			this_scores.append(ys[idx]) # TODO (LDERY) -- double check if this is better in the long run. Initial experiments said no -np.log(-ys[idx]))
	xs, ys = this_masks, this_scores
	print('Total runs = {}, Total dropped = {}'.format(len(is_valid), len(is_valid) - sum(is_valid)))

	sm_hp_searcher = ScoreModelHP(
			id_='{}/{}'.format(parent_id, "Global"), num_players=xs[0].numel(),
			base_mask=torch.zeros_like(xs[0]).view(-1, 1), hp_dict=hp_dict, wandb=wandb_run)

	sm_hp_searcher.search_best_linear_fit((xs, ys))
	best_fit = sm_hp_searcher.get_best_fit()
	score_map[model_type] = best_fit

	return score_map

def updated_run_data_to_sampling_proba(info, module, pfrac):
	if isinstance(info['in'][1], tuple):
		ins_mean, ins_sq = info['in'][1][0] / info['in'][0], info['in'][1][1] / info['in'][0]
		avg_act_magnitudes = (ins_sq - ins_mean**2)
		if hasattr(module, 'num_key_value_heads'):
			avg_act_magnitudes = avg_act_magnitudes.view(1, 1, module.num_heads, -1).mean(axis=-1, keepdim=True)
	else:
		avg_act_magnitudes = info['in'][1] / info['in'][0]
	sampling_proba = avg_act_magnitudes.cpu().squeeze().numpy()
	sampling_proba = (sampling_proba - np.mean(sampling_proba)) / np.std(sampling_proba)

	# hard coded to look at 2x the original pruning fraction
	num_keep_static = 0 if pfrac is None else int(len(sampling_proba)*(1.0 - 2*pfrac))
	sorted_ = np.argsort(-sampling_proba)
	fixed_indices, use_indices = sorted_[:num_keep_static], sorted_[num_keep_static:]

	assert module.main_mask is None, "the main masks of all modules should be none by this point."

	return sampling_proba, fixed_indices, use_indices

def investigate_score_based_mask(args, model, wandb_run, dataset, data_for_prior, tokenizer, epoch_=1):

	def update_mask_one_layer(module, info, score_info, prune_frac, regression_weights, fixed_indices, use_indices, preset_qt=None):
		if isinstance(info['in'][1], tuple):
			if hasattr(module, 'num_key_value_heads'):
				shape_template =(1, 1, module.num_heads, 1)
				score_model_weights = torch.zeros((module.num_heads, )).squeeze().cuda()
			else:
				score_model_weights = torch.zeros_like(info['in'][1][0]).squeeze()
				shape_template = info['in'][1][0].shape
		else:
			score_model_weights = torch.zeros_like(info['in'][1]).squeeze()
			shape_template = info['in'][1].shape

		if regression_weights is None:
			regression_weights = (info['in'][1] / info['in'][0]).squeeze()
			regression_weights = regression_weights[use_indices]

		# bias this so that we do not remove any of the fixed indices
		score_model_weights[fixed_indices] = INF
		score_model_weights[use_indices] = regression_weights

		if preset_qt is None:
			if module.main_mask is not None:
				qt = torch.quantile((score_model_weights[(module.main_mask).squeeze() > 0]).squeeze().float(), prune_frac)
			else:
				qt = torch.quantile(score_model_weights.squeeze().float(), prune_frac)
		else:
			qt = preset_qt

		mask_ = ((score_model_weights > qt)*1.0).half()
		if module.main_mask is not None:
			module.main_mask *= (mask_).view(shape_template)
		else:
			module.main_mask = (mask_).view(shape_template)
		return module.main_mask.mean().item()

	def compute_updated_masks_local(prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=1.0, preset_qt=None, no_regression=False):
		avgs = 0.0
		for id_, (name, module) in module_map.items():
			this_prune_frac = prune_frac
			if name.endswith('self_attn'):
				this_prune_frac = prune_frac * mlp_attn_ratio

			_, fixed_indices, use_indices = all_sampling_proba[id_]
			score_model = None if ((score_model_maps is None) or (no_regression)) else score_model_maps[id_]
			score_matrix_entry = score_matrix[id_] if score_matrix is not None else None
			this_avg = update_mask_one_layer(
										module, info_cache[name], 
										score_matrix_entry, this_prune_frac,
										score_model, fixed_indices, use_indices, preset_qt=preset_qt)
			avgs += this_avg

		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			info_cache[k] = dict()

	def compute_updated_masks_global(prune_frac, score_matrix, score_model_map, all_sampling_proba, mlp_attn_ratio=1.0,):
		start_idx = 0
		for id_, (name, module) in module_map.items():
			this_group = best_fit[start_idx: (start_idx + len(score_perfs[0][id_][0]))]
			score_map[id_] = this_group


	# add forward hooks
	module_map = {}
	info_cache, hook_handles = defaultdict(dict), []
	for (name, module) in model.named_modules():
		# For now, only focus on the MLPs
		if  name.endswith('self_attn') and (args.mlp_attn_ratio == 0):
			continue
		if name.endswith('mlp') or name.endswith('self_attn'):
			# This module has already been fully pruned.
			if module.skip_computation:
				continue

			hook_handles.append(module.register_forward_hook(hook_fn(name, info_cache)))
			id_  = '{}.{}'.format('self_attn' if name.endswith('self_attn') else 'mlp', int(name.split('.')[2]))
			module_map[id_] = (name, module)
			intermediate_sz = module.intermediate_size
			# set the module prune type here
			module.prune_method = args.prune_method

	# Initial setup to get the initial probability distribution for sampling
	get_train_ppl_multitry(model, data_for_prior, args.bsz)
	print('Done generating prior')

	hp_dict = get_linearmodel_hpdict(args)
	score_matrix = defaultdict(lambda: None)
	all_sampling_proba = defaultdict(lambda: np.ones((intermediate_sz)))
	for id_, (name, module) in module_map.items():
		this_pfrac = args.prune_frac
		if name.endswith('self_attn'):
			this_pfrac = this_pfrac * args.mlp_attn_ratio
		this_pfrac = None if args.no_perturb else this_pfrac
		all_sampling_proba[id_] = updated_run_data_to_sampling_proba(info_cache[name], module, this_pfrac)
		if isinstance(info_cache[name]['in'][1], tuple):
			module.main_mask = torch.ones_like(info_cache[name]['in'][1][0]).half()
			if hasattr(module, 'num_key_value_heads'):
				module.main_mask = module.main_mask.view(1, 1, module.num_heads, -1).mean(axis=-1, keepdim=True)
		else:
			module.main_mask = torch.ones_like(info_cache[name]['in'][1]).half()
		module.prune_method = None # we are turning off gathering any pruning statistics

	for handle in hook_handles:
		handle.remove()

	# Do a global sampling here
	## Aggregate all the sampling probabilities
	global_proba = []
	for k, v in all_sampling_proba.items():
		sampling_proba, _, use_indices = v
		global_proba.append(sampling_proba[use_indices])
	global_proba = np.concatenate(global_proba)

	if not args.no_perturb: # We are not running a perturbation algorithm
		global_proba = global_proba.max() - global_proba # Make positive
		# This will be the full distribution amongst all entities.
		global_proba /= np.sum(global_proba)

		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			if isinstance(info_cache[k]['in'][1], tuple):
				info_cache[k]['in'][1][0].zero_()
				info_cache[k]['in'][1][1].zero_()
			else:
				info_cache[k]['in'][1].zero_()

		start = time()
		score_info = get_random_mask_scores(
							model, dataset, module_map, all_sampling_proba, global_proba,
							bsz=args.bsz, nsamples=args.nsamples, use_complement=args.use_complement,
							mpi=args.masks_per_iter, pfrac=args.prune_frac, mlp_attn_ratio=args.mlp_attn_ratio
		)
		gen_scores_time = time() - start
		start = time()
		score_model_maps = get_score_models(score_info, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='Iter.{}'.format(epoch_), model_type=args.sm_lin_model_type)
		time_delta = time() - start
	else:
		gen_scores_time = 0
		time_delta = 0
		score_model_maps = None
		score_matrix = None


	# Need to do some fitting to a linear model here.
	preset_qt = None
	if args.sm_lin_model_type == 'global' and (args.sm_nepochs > 0): # and (not args.no_perturb):
		if args.no_perturb:
			best_fit = global_proba
		else:
			best_fit = score_model_maps[args.sm_lin_model_type].cpu().numpy()
# 		pdb.set_trace()

		init_param_counts = np.zeros_like(best_fit)
		start_idx = 0
		for id_, (name, module) in module_map.items():
			if args.no_perturb:
				num_entries= len(all_sampling_proba[id_][-1])
			else:
				num_entries =  len(score_info[0][id_][0])

			if name.endswith('mlp'):
				init_param_counts[start_idx: (start_idx + num_entries)] = model.model.params_per_pruned_hidden
			else:
				init_param_counts[start_idx: (start_idx + num_entries)] = model.model.params_per_pruned_head
			start_idx += num_entries

		sort_idxs = np.argsort(best_fit)
		cum_sum_param_counts = np.cumsum(init_param_counts[sort_idxs])
		threshold = int(model.original_param_count * args.prune_frac)
		keep_idxs = (cum_sum_param_counts > threshold) * 1.0
		best_fit = torch.tensor(keep_idxs[np.argsort(sort_idxs)]).float().cuda()
		# We need to do some prep here
		start_idx = 0
		score_model_maps = {}
		for id_, (name, module) in module_map.items():
			if args.no_perturb:
				num_entries= len(all_sampling_proba[id_][-1])
			else:
				num_entries =  len(score_info[0][id_][0])

			this_group = best_fit[start_idx: (start_idx + num_entries)]
			score_model_maps[id_] = this_group
			start_idx += num_entries
		preset_qt = 0

	
	compute_updated_masks_local(args.prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=args.mlp_attn_ratio, preset_qt=preset_qt, no_regression=(args.sm_nepochs == 0))
	if wandb_run is not None:
		wandb_run.log({'SysStats/scoreruntime': gen_scores_time, 'SysStats/pruneruntime': time_delta})
	mask_info = {name: module.main_mask.clone() for _, (name, module) in module_map.items()}

	return mask_info

def args_to_dict(args):
	def stringify(x):
		return '-'.join([str(y) for y in eval(x)])

	return {
		'nsamp': args.nsamples,
		'sp': args.sparsity_ratio,
		'pfrac': args.prune_frac,
		'bsz': args.bsz,
		'mpi': args.masks_per_iter,
		'pmethod': args.prune_method,
		'P-Seqlen': args.prune_seqlen,
		'Lin.regtype': args.sm_reg_type, 
		'Lin.regW': stringify(args.sm_reg_weight),
		'Lin.lr': stringify(args.sm_lr_factor),
		'Lin.bsz': stringify(args.sm_bsz),
		'Lin.neps': args.sm_nepochs,
		'Lin.type': args.sm_lin_model_type,
		'name': args.wandb_project_name,
		'bias_ns': args.bias_ns,
		'prion_ns': args.prior_ns,
		'complement': args.use_complement
	}

def args_to_str(args):
	relevant_args = args_to_dict(args)
	return '_'.join(['{}={}'.format(k, v) for k, v in relevant_args.items()])

def get_linearmodel_hpdict(args):
	base_hp = {
		'lr_factor' : eval(args.sm_lr_factor),
		'reg_weight': eval(args.sm_reg_weight),
		'reg_type': [args.sm_reg_type],
		'bsz' : eval(args.sm_bsz),
		'nepochs' : [args.sm_nepochs],
		'patience': [10],
	}
	return base_hp

def get_param_count(model, exclude=['embed', 'head']):
	return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

def prune_mlp(mask_, module):
	# Reset pruning related information
	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print("We are pruning the whole mlp layer")
		module.gate_proj = None
		module.up_proj   = None
		module.down_proj = None
		module.intermediate_size = 0
		module.skip_computation = True
	else:
		index = mask_.squeeze().nonzero().squeeze()
		new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
		module.gate_proj = None
		module.gate_proj = new_gate_proj
		new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
		module.up_proj  = None
		module.up_proj = new_up_proj
		new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
		module.down_proj = None
		module.down_proj = new_down_proj
		module.intermediate_size = len(index)

def prune_attn(mask_, module):

	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print('We are pruning a whole attention layer')
		module.q_proj = None
		module.k_proj = None
		module.v_proj = None
		module.o_proj = None
		module.skip_computation = True
		module.num_heads = 0
		module.hidden_size = 0
		module.intermediate_size = 0
	else:
		index = (mask_.squeeze() == 0).nonzero().squeeze()
		if index.numel() == 1:
			index = [index]

		_, updated_indices = find_pruneable_heads_and_indices(
			index, module.num_heads, module.head_dim, set()
		)

		new_q_proj = (prune_linear_layer(module.q_proj, updated_indices)).half()
		module.q_proj = None
		module.q_proj = new_q_proj

		new_k_proj = (prune_linear_layer(module.k_proj, updated_indices)).half()
		module.k_proj = None
		module.k_proj = new_k_proj

		new_v_proj = (prune_linear_layer(module.v_proj, updated_indices)).half()
		module.v_proj = None
		module.v_proj = new_v_proj

		new_o_proj = (prune_linear_layer(module.o_proj, updated_indices, dim=1)).half()
		module.o_proj = None
		module.o_proj = new_o_proj

		module.num_heads = len(mask_.squeeze().nonzero())
		module.num_key_value_heads = module.num_heads
		module.hidden_size = module.num_heads * module.head_dim
		module.intermediate_size = module.num_heads


def prune_model(args, model, mask_info, tokenizer, bias_calibration_data, bias_info=None, epoch=1):
	info_cache, hook_handles = defaultdict(dict), []
	if 'bias' in args.repair_method:
		for (name, module) in model.named_modules():
			if name not in mask_info: continue # We are not pruning this

			# Reset to the original before running
			module.main_mask = None
			module.temp_mask = None
			module.computing_updated_bias = 1 - mask_info[name]
			hook_handles.append(module.register_forward_hook(hook_fn(name, info_cache)))

		# do garbage collection here
		gc.collect()
		torch.cuda.empty_cache()

		this_ppl, _ = get_train_ppl_multitry(model, bias_calibration_data, args.bsz)
		for handle in hook_handles:
			handle.remove()

	for (name, module) in model.named_modules():
		if name not in mask_info: continue # We are not pruning this

		mask_ = mask_info[name]
		if 'bias' in args.repair_method:
			new_param = (info_cache[name]['in'][1]/(info_cache[name]['in'][0] * epoch)).squeeze()
			if bias_info[name] is not None:
				bias_info[name].mul_((epoch - 1)/epoch).add_(new_param)
			else:
				bias_info[name] = new_param
		if name.endswith('mlp'):
			prune_mlp(mask_, module)
		elif name.endswith('self_attn') and (args.mlp_attn_ratio != 0):
			prune_attn(mask_, module)
		else:
			raise ValueError("Invalid type found in mask_info : {}".format(name))

		del new_param
		module.computing_updated_bias = None

	gc.collect()
	torch.cuda.empty_cache()



def post_pruning_bias_fix(model, bias_info):
	for name, module in model.named_modules():
		if name.endswith('self_attn'):
			device = module.o_proj.weight.device
			module.o_proj.bias = torch.nn.Parameter((bias_info[name]).half().to(device))
		elif name.endswith('mlp'):
			device = module.down_proj.weight.device
			module.down_proj.bias = torch.nn.Parameter((bias_info[name]).half().to(device))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='LLaMA model') # huggyllama/llama-7b
	parser.add_argument('--dataset', type=str, default="wikitext2", choices=["wikitext2", "c4"])
	parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
	parser.add_argument('--nsamples', type=int, default=14, help='Number of calibration samples.')
	parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
	parser.add_argument('--prune_frac', type=float, default=0.1, help='Fraction of weights to prune at a time')
	parser.add_argument('--bsz', type=int, default=14, help='Instantaneous batch size for forward pass')
	parser.add_argument('--mlp_attn_ratio', type=float, default=1.0, help="For a given prune_frac, the ratio of the pruning for attn vrs mlp")

	parser.add_argument('--use_complement', action="store_true", help="Whether to use complement")

	parser.add_argument('--prune_method', type=str, default="magnitude", choices=["magnitude", "wanda", "random", "fluct", "fluct.2.0"])
	parser.add_argument("--cache_dir", default="llm_weights", type=str )
	parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
	parser.add_argument('--save', type=str, default=None, help='Path to save results.')
	parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
	parser.add_argument('--masks_per_iter', type=int, default=10, help='How many masks to generate per-iteration')
	parser.add_argument('--tol', type=float, default=0.02, help="What level of tolerance close to the target sparsity to accept")
	parser.add_argument('--no_perturb', action="store_true", help="We do not perform any perturbation")
	parser.add_argument('--prune_seqlen', type=int, default=-1, help='the sequence length to use for pruning')
	parser.add_argument('--bias_ns', type=int, default=32, help='Number of samples to use when estimating bias')
	parser.add_argument('--prior_ns', type=int, default=32, help='Number of samples to use when estimating prior')

	# Hyperparams for scoring model
	parser.add_argument('--sm_reg_weight', type=str, default='[1e2, 1e-4, 0]', help='reg-weight to use')
	parser.add_argument('--sm_lr_factor', type=str, default='[100, 10, 1, 0.1]', help='lr factor to use for fitting linear model')
	parser.add_argument('--sm_reg_type', type=str, default="l1", help='type of regularization to apply')
	parser.add_argument('--sm_lin_model_type', type=str, default="global", help='type of regularization to apply') 


	parser.add_argument('--sm_bsz', type=str, default='[32, 64, 128]', help='batch size for fitting linear model')
	parser.add_argument('--sm_nepochs', type=int, default=50, help='number of epochs to use to fit the linear model')
	parser.add_argument('--last-epoch', type=int, default=-1)
	parser.add_argument('--repair_method', type=str, default='bias', choices=["none", "bias"])

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
	model = get_llm(args.model, args.cache_dir, args.prune_seqlen)
	if args.prune_seqlen < 0:
		args.prune_seqlen = model.config.max_position_embeddings # set seqlen to the model seqlen
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=('olmo' in args.model.lower()))

	model.seqlen = model.config.max_position_embeddings # set seqlen to the model seqlen for evaluation
	# Get the test loader
	trainloader, testloader = get_loaders(
		args.dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	start_time = time()
	orig_train_ppl, orig_test_ppl = -1, -1 #eval_ppl(model, trainloader, testloader, model.device, bsz=args.bsz)
	model.seqlen = args.prune_seqlen
	original_runtime = time() - start_time
	print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(0.0, orig_train_ppl, orig_test_ppl))
	print('The model seqlen is set to ',model.seqlen)

	bias_info, bias_calibration_data = None, None
	if args.repair_method == 'bias':
		bias_info = defaultdict(lambda: None)
		bias_calibration_data, _ = get_loaders(
			args.dataset, nsamples=args.bias_ns, seed=random.randint(0, 9999), seqlen=model.seqlen, tokenizer=tokenizer 
		)
		print("Done loading bias data")

	# Get the full dataset that we are going to be passing around!
	mul_factor = int((args.sparsity_ratio / args.prune_frac)) + 4 # + 4 is a buffer in case the pruning ends up needing more steps
	total_samples = args.nsamples * mul_factor
	prior_total_samples = args.prior_ns * mul_factor
	full_dataset, _ = get_loaders(
			args.dataset, nsamples=(total_samples + prior_total_samples), seed=random.randint(0, 9999), seqlen=model.seqlen, tokenizer=tokenizer 
	)
	full_dataset, prior_dataset = full_dataset[:total_samples], full_dataset[total_samples:]
	print("Done loading prior and perturbation data")

	original_param_count = get_param_count(model)
	model.original_param_count = original_param_count
	cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
	epoch_, prune_runtimes = 1, []
	while True:
		if (abs(cur_sparsity - args.sparsity_ratio) < args.tol) or (cur_sparsity > args.sparsity_ratio):
			break

		# Need to check if we have to clip the sparsity ratio
		if (cur_sparsity + args.prune_frac) > args.sparsity_ratio:
			# We would overshoot in this case which is not idea.
			old_prune_frac = args.prune_frac
			args.prune_frac = abs(args.sparsity_ratio - cur_sparsity)
			print('We have updated the prune fraction {:.3f} -> {:.3f} to avoid overshooting'.format(old_prune_frac, args.prune_frac))


		print('Gathering statistics for pruning')
		save_loc = os.path.join(args.save, 'mask_info_{}.pkl'.format(epoch_))
		if os.path.exists(save_loc):
			print('Successfully loaded past pruning info')
			with open(save_loc, 'rb') as handle:
				mask_info = pkl.load(handle)
		else:
			this_data = full_dataset[(args.nsamples * (epoch_ - 1)):(args.nsamples * epoch_)]
			this_prior = prior_dataset[(args.prior_ns * (epoch_ - 1)):(args.prior_ns * epoch_)]
			start_ = time()
			mask_info = investigate_score_based_mask(args, model, wandb_run, this_data, this_prior, tokenizer, epoch_=epoch_)
			prune_runtimes.append(time() - start_)
			# Save the mask info for the epoch
			with open(save_loc, 'wb') as handle:
				pkl.dump(mask_info, handle)

		print('Prune model')
		prune_model(args, model, mask_info, tokenizer, bias_calibration_data, bias_info, epoch=epoch_)
		cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
		pprint({k: v.shape for k, v in model.named_parameters() if ('o_proj' in k) or ('down_proj' in k)})

		start_time = time()
		model.seqlen = model.config.max_position_embeddings # set seqlen to the model seqlen for evaluation
		ppl_train, ppl_test = eval_ppl(model, trainloader, testloader, model.device, bsz=4)
		model.seqlen = args.prune_seqlen # reset the seqlen for pruning
		pruned_model_runtime = time() - start_time

		if wandb_run is not None:
			wandb_run.log({
				'Relative-Speedup': original_runtime / pruned_model_runtime,
			})

			wandb_run.log({'Sparsity': cur_sparsity, 'TrainPPL': ppl_train, 'TestPPL': ppl_test})
		print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(cur_sparsity, ppl_train, ppl_test))

		epoch_ += 1
		if epoch_ == args.last_epoch:
			break

	print('Pruning took {:.3f} min on average'.format(np.mean(prune_runtimes) / 60.0))
	if args.repair_method == 'bias':
		post_pruning_bias_fix(model, bias_info)
		model.seqlen = model.config.max_position_embeddings # set seqlen to the model seqlen for evaluation
		ppl_train, ppl_test = eval_ppl(model, trainloader, testloader, model.device, bsz=4)
		model.seqlen = args.prune_seqlen # reset the seqlen for pruning
		print('Post-Bias-Fix-Train PPL = {:.3f} | Post-Bias-Fix-Test PPL = {:.3f}'.format(ppl_train, ppl_test))
		if wandb_run is not None:
			wandb_run.log({'Post-Bias-Fix-TrainPPL': ppl_train, 'Post-Bias-Fix-TestPPL': ppl_test})
			wandb_run.log({'sparsity': cur_sparsity})

import cProfile
import pstats
if __name__ == '__main__':
# 	with cProfile.Profile() as profile_ctxt:
	main()
# 	pdb.set_trace()
# 	ps = pstats.Stats(profile_ctxt).strip_dirs().sort_stats('tottime')
# 	ps.print_stats(20)
# 	ps.print_caller(20)
# 	pdb.set_trace()
# 	ps.print_stats()