#!/bin/bash


echo "Running for Magnitude..."
python my_main.py --model mistralai/Mistral-7B-v0.1 --save out --masks_per_iter 200 --bsz 16 --prune_frac 0.05 --nsamples 32 && 
	
	

python my_main.py --model mistralai/Mistral-7B-v0.1 --save out --masks_per_iter 200 --bsz 16 --prune_method wanda --prune_frac 0.05 --nsamples 32

echo "Running Wanda..."

echo "Both were successful!"
