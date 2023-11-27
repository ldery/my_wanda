# Import necessary modules
import time
import torch
import torch.nn as nn
import pdb
# Import get_loaders function from data module within the same directory
from .data import get_loaders
from datasets import load_dataset
from torch.optim import AdamW
import numpy as np
import gc

def wikitext_train_epoch(model, trainloader, optimizer, device=torch.device("cuda:0"), bs=8):
	trainloader = trainloader.input_ids
	# TODO (ldery) -- fix the number of samples
	nsamples = 20 #trainloader.shape[-1]
	train_losses = []
	print('This is the model sequence length: ', model.seqlen)
	max_seq_len = model.seqlen
	# TODO -- ldery -- shuffle the data at some point
	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		j = min(i+bs, nsamples)

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
		optimizer.step()
		optimizer.zero_grad()
		train_losses.append(loss.item())

	# Empty CUDA cache to save memory
	gc.collect()
	torch.cuda.empty_cache()
	return np.mean(train_losses)

def train_wikitext(model, tokenizer, device=torch.device("cuda:0"), seed=0, lr_=1e-5):
	# Set dataset
	dataset = "wikitext2"

	traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
	validdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
	testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

	# Encode datasets
	trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
	validenc = tokenizer(" ".join(validdata['text']), return_tensors='pt')
	testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

	
	optimizer = AdamW(model.parameters(), lr=lr_)
	best_ppl_so_far, final_test_ppl = float('inf'), float('inf')
	for epoch_ in range(10):
		model.train()
		train_loss = wikitext_train_epoch(model, trainenc, optimizer, bs=1, device=device)
		print(epoch_, train_loss)
		model.eval()
		with torch.no_grad():
			valid_ppl = eval_ppl_wikitext(model, validenc, bs=8, device=device)
			test_ppl = eval_ppl_wikitext(model, testenc, bs=8, device=device)
		model.train()
		if valid_ppl < best_ppl_so_far:
			best_ppl_so_far = valid_ppl
			final_test_ppl = test_ppl
		torch.cuda.empty_cache()
		print('Train Loss = {:.3f} | Val PPL Best = {:.3f} | Test PPL {:.3f}'.format(train_loss, best_ppl_so_far, test_ppl))
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