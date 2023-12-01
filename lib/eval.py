# Import necessary modules
import time
import torch
import torch.nn as nn
import pdb
# Import get_loaders function from data module within the same directory
from .data import get_loaders
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gc

def wikitext_train_epoch(model, trainloader, optimizer, training_hp, device=torch.device("cuda:0")):
	trainloader = trainloader.input_ids
	nsamples = training_hp['ft_steps'] *  training_hp['bsz'] * training_hp['grad_accum_steps']
	train_losses = []
	print('This is the model sequence length: ', model.seqlen)

	# TODO (ldery) -- fix the max sequence lenght
	max_seq_len = int(model.seqlen * 0.9)
	# TODO -- ldery -- shuffle the data at some point
	# Loop through each batch
	for i in range(0, nsamples, training_hp['bsz']):
		if i % (training_hp['grad_accum_steps'] * 10) == 0:
			print('On step : ', i // training_hp['grad_accum_steps'])

		# Calculate end index
		j = min(i + training_hp['bsz'], nsamples)

		# Prepare inputs and move to device
		inputs = trainloader[:,(i * max_seq_len):(j * max_seq_len)].to(device)
		inputs = inputs.reshape(j-i,max_seq_len)

		# Forward pass through the model
		lm_logits = model(inputs).logits

		# Shift logits and labels for next token prediction
		shift_logits = lm_logits[:, :-1, :].contiguous()
		shift_labels = inputs[:, 1:]

		# Compute loss
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		loss.backward()
		if i % training_hp['grad_accum_steps'] == 0:
			optimizer.step()
			optimizer.zero_grad()
		train_losses.append(loss.item())

	# Empty CUDA cache to save memory
	gc.collect()
	torch.cuda.empty_cache()
	return np.mean(train_losses)

def train_wikitext(model, tokenizer, trainable_params, training_hp, wandb=None, device=torch.device("cuda:0"), seed=0):
	# Set dataset
	dataset = "wikitext2"

	traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
	validdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
	testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

	# Encode datasets
	trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
	validenc = tokenizer(" ".join(validdata['text']), return_tensors='pt')
	testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

	
	optimizer = AdamW(trainable_params, lr=training_hp['lr'])
	lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)

	best_ppl_so_far, final_test_ppl = float('inf'), float('inf')
	no_improvement_so_far = 0
	for epoch_ in range(10):
		model.train()
		train_loss = wikitext_train_epoch(model, trainenc, optimizer, training_hp, device=device)
		print(epoch_, train_loss)
		model.eval()
		with torch.no_grad():
			valid_ppl = eval_ppl_wikitext(model, validenc, bs=8, device=device)
			test_ppl = eval_ppl_wikitext(model, testenc, bs=8, device=device)
		model.train()

		lr_scheduler.step(valid_ppl)
		if valid_ppl < best_ppl_so_far:
			best_ppl_so_far = valid_ppl
			final_test_ppl = test_ppl
			no_improvement_so_far = 0
		gc.collect()
		torch.cuda.empty_cache()
		print('[Epoch {}] Train Loss = {:.3f} | Val PPL Best = {:.3f} | Test PPL {:.3f}'.format(epoch_, train_loss, best_ppl_so_far, test_ppl))
		if wandb is not None:
			wandb.log({'FT.valid_ppl': best_ppl_so_far, 'FT.TestPPL': test_ppl})

		no_improvement_so_far += 1
		if no_improvement_so_far > 3: # have a patience of 3
			break

	if wandb is not None:
		wandb.log({'FT.FinalPPL': test_ppl})
	return best_ppl_so_far, final_test_ppl

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
	# Set dataset
	dataset = "wikitext2"

	# Print status
	print(f"evaluating on {dataset}")

	# Get the test loader
	trainloader, testloader = get_loaders(
		dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
		ppl_train = eval_ppl_wikitext_train(model, trainloader, 1, device)
	return ppl_train, ppl_test 

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl_trainonly(model, tokenizer, bsz=1, nsamples=128, device=torch.device("cuda:0"), seed=0):
	# Set dataset
	dataset = "wikitext2"

	# Get the test loader
	trainloader, _ = get_loaders(
		dataset, nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		ppl_train = eval_ppl_wikitext_train(model, trainloader, bsz, device)
	return ppl_train

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
	# Get input IDs
	# testenc = testenc.input_ids

	# Calculate number of samples
	# nsamples = testenc.numel() // model.seqlen
	nsamples = len(trainloader)

	# List to store negative log likelihoods
	nlls = []
	print(f"nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		j = min(i+bs, nsamples)

		# Prepare inputs and move to device
		# inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
		this_bs = min(bs, nsamples - i)
		inputs = torch.concat([trainloader[i + k][0].to(device) for k in range(this_bs)])

		inputs = inputs.reshape(j-i, model.seqlen)

		# Forward pass through the model
		lm_logits = model(inputs).logits

		# Shift logits and labels for next token prediction
		shift_logits = lm_logits[:, :-1, :].contiguous()
		shift_labels = inputs[:, 1:]

		# Compute loss
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		# Calculate negative log likelihood
		neg_log_likelihood = loss.float() * model.seqlen * (j-i)

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
	# Get input IDs
	testenc = testenc.input_ids

	# Calculate number of samples
	nsamples = testenc.numel() // model.seqlen

	# List to store negative log likelihoods
	nlls = []
	print(f"nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		j = min(i+bs, nsamples)

		# Prepare inputs and move to device
		inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
		inputs = inputs.reshape(j-i, model.seqlen)

		# Forward pass through the model
		lm_logits = model(inputs).logits

		# Shift logits and labels for next token prediction
		shift_logits = lm_logits[:, :-1, :].contiguous()
		shift_labels = inputs[:, 1:]

		# Compute loss
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		# Calculate negative log likelihood
		neg_log_likelihood = loss.float() * model.seqlen * (j-i)

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()