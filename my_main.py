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
from lib.scoring_model import ScoreModelHP
import wandb
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import gc

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

INF = 1e8

def set_masks(module_map, all_masks, all_sampling_proba, pfrac=0.1, mlp_attn_ratio=1.0, use_complement=False):
	for k, (name, module) in module_map.items():
		if name.endswith('self_attn'):
			this_pfrac = pfrac * mlp_attn_ratio

		module.is_using_main = False
		sampling_proba, fixed_indices, use_indices = all_sampling_proba[k]
		if use_complement:
			module.temp_mask = 1 - module.temp_mask
			module.temp_mask[:, :, fixed_indices] = 1.0
			all_masks[k].append(module.temp_mask.cpu().squeeze()[use_indices])
		else:
			mask = get_random_mask(module.main_mask.numel(), module.main_mask, sampling_proba, this_pfrac)
			module.temp_mask = torch.Tensor(mask).type(module.main_mask.type())
			all_masks[k].append(torch.Tensor(mask).squeeze()[use_indices])

def get_random_mask(intermediate_sz, main_mask, sampling_proba, pfrac):
	init_set = np.ones((1, 1, intermediate_sz)) if main_mask is None else main_mask.cpu().numpy()
	num_to_zero = int(pfrac * np.sum(init_set)) + 1
	non_zero_idxs = np.squeeze(init_set).nonzero()[0]
	new_proba = sampling_proba[non_zero_idxs]
	new_proba = new_proba / np.sum(new_proba)
	chosen_idxs = np.random.choice(non_zero_idxs, size=num_to_zero, p=new_proba)
	init_set[:, :, chosen_idxs] = 0
	return init_set

def get_random_mask_scores(model, tokenizer, module_map, all_sampling_proba, bsz=12, nsamples=32, mpi=100, pfrac=0.1, mlp_attn_ratio=1.0):

	# set to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = False

	all_masks, all_perfs = defaultdict(list), defaultdict(list)
	seed_ = random.randint(0, 1e4)
	for iter_ in range(mpi // 2):
		this_bsz = bsz

		# set the layer mask here
		set_masks(module_map, all_masks, all_sampling_proba, pfrac=pfrac, mlp_attn_ratio=mlp_attn_ratio)

		# TODO (clean up with a wrapper since this is repeated code)
		try:
			this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_)
		except Exception as e:
			print(e)
			gc.collect()
			torch.cuda.empty_cache()
			this_bsz = 1
			this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_)

		print('[v1]Iter : ', iter_, ' PPL = ', this_ppl)
		this_ppl = this_ppl if this_ppl < INF else INF

		for k, (name, module) in module_map.items():
			# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
			all_perfs[k].append(-this_ppl)

		# set the complement mask here
		set_masks(module_map, all_masks, all_sampling_proba, pfrac=pfrac, mlp_attn_ratio=mlp_attn_ratio, use_complement=True)
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_)

		print('[v2]Iter : ', iter_, ' PPL = ', this_ppl)
		this_ppl = this_ppl if this_ppl < INF else INF

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
		flat_in = (module.intermed_cache).clone().detach().float()
		module.intermed_cache = None
		if 'in' not in info_cache[module_name]:
			info_cache[module_name]['in'] = [1, flat_in]
		else:
			info_cache[module_name]['in'] = [
				info_cache[module_name]['in'][0] + 1,  
				info_cache[module_name]['in'][1].add_(flat_in)
			]
	return hook

def get_score_models(score_perfs, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='.',  model_type='local'):
	score_map = {}
	if model_type == 'local':
		for id_, (name, module) in module_map.items():
			# Get a score map
			_, _, use_indices = all_sampling_proba[id_]
			base_mask = info_cache[name]['in'][1] / info_cache[name]['in'][0]
			base_mask = (base_mask.squeeze() * module.main_mask.squeeze().float())[use_indices]
			base_mask = (base_mask / base_mask.sum()).view(-1, 1)

			sm_hp_searcher = ScoreModelHP(
				id_='{}/{}'.format(parent_id, id_), num_players=score_perfs[0][id_][0].numel(),
				base_mask=base_mask, hp_dict=hp_dict, wandb=wandb_run)

			run_info = score_perfs[0][id_], score_perfs[1][id_]

			sm_hp_searcher.search_best_linear_fit(run_info)
			score_map[id_] = sm_hp_searcher.get_best_fit()
	else:
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
		sm_hp_searcher = ScoreModelHP(
				id_='{}/{}'.format(parent_id, "Global"), num_players=xs[0].numel(),
				base_mask=torch.zeros_like(xs[0]).view(-1, 1), hp_dict=hp_dict, wandb=wandb_run)

		sm_hp_searcher.search_best_linear_fit((xs, ys))
		best_fit = sm_hp_searcher.get_best_fit()
		start_idx = 0
		for id_, (name, module) in module_map.items():
			this_group = best_fit[start_idx: (start_idx + len(score_perfs[0][id_][0]))]
			score_map[id_] = this_group
	return score_map

def run_data_to_sampling_proba(info, module, pfrac):
	avg_act_magnitudes = info['in'][1] / info['in'][0]
	sampling_proba = avg_act_magnitudes.cpu().squeeze().numpy()

	num_keep_static = int(len(sampling_proba)*(1.0 - 2*pfrac)) # hard coded to look at 2x the original pruning fraction
	sorted_ = np.argsort(-sampling_proba)
	fixed_indices, use_indices = sorted_[:num_keep_static], sorted_[num_keep_static:]

	sampling_proba = sampling_proba.max() - sampling_proba
	sampling_proba[fixed_indices] = 0

	if module.main_mask is not None:
		sampling_proba *= (module.main_mask).cpu().float().squeeze().numpy()
	sampling_proba /= np.sum(sampling_proba)
	
	if np.isnan(sampling_proba).any():
		print('We got nan in the sampling probability')
		pdb.set_trace()

	assert not np.isnan(sampling_proba).any(), 'Nans encountered in the sampling probability distribution'
	return sampling_proba, fixed_indices, use_indices

def investigate_score_based_mask(args, model, wandb_run, epoch_=1):

	def update_mask_one_layer(module, info, score_info, prune_frac, regression_weights, fixed_indices, use_indices):
		score_model_weights = torch.zeros_like(info['in'][1]).squeeze()
		if regression_weights is None:
			regression_weights = (info['in'][1] / info['in'][0]).squeeze()

		# bias this so that we do not remove any of the fixed indices
		score_model_weights[fixed_indices] = regression_weights.max()
		score_model_weights[use_indices] = regression_weights


		if module.main_mask is not None:
			qt = torch.quantile((score_model_weights[(module.main_mask).squeeze() > 0]).squeeze().float(), prune_frac)
		else:
			qt = torch.quantile(score_model_weights.squeeze().float(), prune_frac)

		mask_ = ((score_model_weights > qt)*1.0).half()

		if mask_.sum() == 0:
			print('We have encountered a zero sum. Need to fix. Skip Updating this layer')
			sampling_proba = run_data_to_sampling_proba(info, module, prune_frac)
			return module.main_mask.mean().item()

		if module.main_mask is not None:
			module.main_mask *= (mask_).view(info['in'][1].shape)
		else:
			module.main_mask = (mask_).view(info['in'][1].shape)

		return module.main_mask.mean().item()

	def compute_updated_masks_local(prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=1.0):
		avgs = 0.0
		for id_, (name, module) in module_map.items():
			this_prune_frac = prune_frac
			if name.endswith('self_attn'):
				this_prune_frac = prune_frac * mlp_attn_ratio

			_, fixed_indices, use_indices = all_sampling_proba[id_]
			score_model = None if score_model_maps is None else score_model_maps[id_]
			this_avg = update_mask_one_layer(
										module, info_cache[name], 
										score_matrix[id_], this_prune_frac,
										score_model, fixed_indices, use_indices)
			avgs += this_avg

		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			info_cache[k] = dict()

	# add forward hooks
	module_map = {}
	info_cache, hook_handles = defaultdict(dict), []
	for (name, module) in model.named_modules():
		# For now, only focus on the MLPs
		if name.endswith('mlp') or name.endswith('self_attn'):
			hook_handles.append(module.register_forward_hook(hook_fn(name, info_cache)))
			id_  = '{}.{}'.format('self_attn' if name.endswith('self_attn') else 'mlp', int(name.split('.')[2]))
			module_map[id_] = (name, module)
			intermediate_sz = module.intermediate_size
			# set the module prune type here
			module.prune_method = args.prune_method

	# Initial setup to get the initial probability distribution for sampling
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

	try:
		eval_ppl_trainonly(model, tokenizer, bsz=args.bsz, nsamples=args.nsamples)
	except Exception as e:
		print(e)
		gc.collect()
		torch.cuda.empty_cache()
		eval_ppl_trainonly(model, tokenizer, bsz=1, nsamples=args.nsamples)

	hp_dict = get_linearmodel_hpdict(args)
	score_matrix = defaultdict(lambda: None)
	all_sampling_proba = defaultdict(lambda: np.ones((intermediate_sz)))
	for id_, (name, module) in module_map.items():
		this_pfrac = args.prune_frac
		if name.endswith('self_attn'):
			this_pfrac = this_pfrac * args.mlp_attn_ratio
		all_sampling_proba[id_] = run_data_to_sampling_proba(info_cache[name], module, this_pfrac)
		module.main_mask = torch.ones_like(info_cache[name]['in'][1]).half()

	# Clear the info-cache for the next round !
	for k, v in info_cache.items():
		info_cache[k] = dict()

	start = time()
	score_info = get_random_mask_scores(
						model, tokenizer, module_map, all_sampling_proba,
						bsz=args.bsz, nsamples=args.nsamples,
						mpi=args.masks_per_iter, pfrac=args.prune_frac, mlp_attn_ratio=args.mlp_attn_ratio
	)
	score_model_maps = get_score_models(score_info, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='Iter.{}'.format(epoch_), model_type=args.sm_lin_model_type)
	gen_scores_time = time() - start
	start = time()
	# Need to do some fitting to a linear model here.
	compute_updated_masks_local(args.prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=args.mlp_attn_ratio)
	time_delta = time() - start
	wandb_run.log({'SysStats/scoreruntime': gen_scores_time, 'SysStats/pruneruntime': time_delta})
	mask_info = {name: module.main_mask for _, (name, module) in module_map.items()}

	for handle in hook_handles:
		handle.remove()

	return mask_info

def args_to_dict(args):
	def stringify(x):
		return '-'.join([str(y) for y in eval(x)])

	return {
		'nsamp': args.nsamples,
		'sp': args.sparsity_ratio,
		'pfrac': args.prune_frac,
		'bsz': args.bsz,
		'ma_ratio': args.mlp_attn_ratio,
		'mpi': args.masks_per_iter,
		'Lin.regtype': args.sm_reg_type, 
		'pmethod': args.prune_method,
		'mlp_attn_ratio': args.mlp_attn_ratio,
		'Lin.regweight': stringify(args.sm_reg_weight),
		'Lin.lr': stringify(args.sm_lr_factor),
		'Lin.bsz': stringify(args.sm_bsz),
		'Lin.nepochs': args.sm_nepochs,
		'Lin.type': args.sm_lin_model_type,
		'name': args.wandb_project_name
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
	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.intermediate_size = len(index)

	if len(index) == 0:
		print("We are pruning the whole layer mlp")
		# alert to skip computation for this module
		module.skip_computation = True

	module.ins_ = None
	gc.collect()
	torch.cuda.empty_cache()

def prune_attn(mask_, module):
	index = (mask_.squeeze() == 0).nonzero().squeeze()

	if index.numel() < 2:
		if index.numel() == 0: return # we are not pruning anything here
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
	module.hidden_size = module.num_heads * module.head_dim

	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.intermediate_size = module.num_heads
	module.ins_ = None

	if len(updated_indices) == 0:
		print("We are pruning the whole layer attn")
		# alert to skip computation for this module
		module.skip_computation = True

	gc.collect()
	torch.cuda.empty_cache() 


def prune_model(args, model, mask_info, tokenizer):
	for (name, module) in model.named_modules():
		if name not in mask_info: continue # We are not pruning this

		mask_ = mask_info[name]
		if name.endswith('mlp'):
			prune_mlp(mask_, module)
		elif name.endswith('self_attn'):
			prune_attn(mask_, module)
		else:
			raise ValueError("Invalid type found in mask_info : {}".format(name))

	gc.collect()
	torch.cuda.empty_cache() 


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='LLaMA model') # huggyllama/llama-7b
	parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
	parser.add_argument('--nsamples', type=int, default=14, help='Number of calibration samples.')
	parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
	parser.add_argument('--prune_frac', type=float, default=0.1, help='Fraction of weights to prune at a time')
	parser.add_argument('--bsz', type=int, default=14, help='Instantaneous batch size for forward pass')
	parser.add_argument('--mlp_attn_ratio', type=float, default=1.0, help="For a given prune_frac, the ratio of the pruning for attn vrs mlp")

	parser.add_argument('--prune_method', type=str, default="magnitude", choices=["magnitude", "wanda"])
	parser.add_argument("--cache_dir", default="llm_weights", type=str )
	parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
	parser.add_argument('--save', type=str, default=None, help='Path to save results.')
	parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
	parser.add_argument('--masks_per_iter', type=int, default=10, help='How many masks to generate per-iteration')
	parser.add_argument('--tol', type=float, default=0.02, help="What level of tolerance close to the target sparsity to accept")

	# Hyperparams for scoring model
	parser.add_argument('--sm_reg_weight', type=str, default='[1e2, 1e-4, 0]', help='reg-weight to use')
	parser.add_argument('--sm_lr_factor', type=str, default='[100, 10, 1, 0.1]', help='lr factor to use for fitting linear model')
	parser.add_argument('--sm_reg_type', type=str, default="l1", help='type of regularization to apply')
	parser.add_argument('--sm_lin_model_type', type=str, default="global", help='type of regularization to apply') 


	parser.add_argument('--sm_bsz', type=str, default='[32, 64, 128]', help='batch size for fitting linear model')
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
	
	start_time = time()
	_, orig_test_ppl = eval_ppl(model, tokenizer, model.device)
	original_runtime = time() - start_time

	origina_param_count = get_param_count(model)
	cur_sparsity = 1.0 - (get_param_count(model) / origina_param_count)
	epoch_ = 1
	while True:
		if (abs(cur_sparsity - args.sparsity_ratio) < args.tol) or (cur_sparsity > args.sparsity_ratio):
			break

		print('Gathering statistics for pruning')
		save_loc = os.path.join(args.save, 'mask_info_{}.pkl'.format(epoch_))
		if os.path.exists(save_loc):
			print('Successfully loaded past pruning info')
			with open(save_loc, 'rb') as handle:
				mask_info = pkl.load(handle)
		else:
			mask_info = investigate_score_based_mask(args, model, wandb_run, epoch_=epoch_)
			# Save the mask info for the epoch
			with open(save_loc, 'wb') as handle:
				pkl.dump(mask_info, handle)

		print('Prune model')
		prune_model(args, model, mask_info, tokenizer) # Do some stuffs here :)
		cur_sparsity = 1.0 - (get_param_count(model) / origina_param_count)

		# Evaluate the performance of the pruned model
		start_time = time()
		ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device)
		pruned_model_runtime = time() - start_time

		wandb_run.log({
			'Relative-Speedup': original_runtime / pruned_model_runtime,
		})

		wandb_run.log({'Sparsity': cur_sparsity, 'TrainPPL': ppl_train, 'TestPPL': ppl_test})
		print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(cur_sparsity, ppl_train, ppl_test))

		epoch_ += 1

	wandb_run.log({'sparsity': cur_sparsity})


if __name__ == '__main__':
    main()
