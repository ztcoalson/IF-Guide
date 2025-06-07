#!/bin/bash

model="pythia-160m"
ckpt_dir="models/training/pythia-160m/baseline/2025-04-11-11-14-04/checkpoint-7628" # example checkpoint path
output_dir="models/training/pythia-160m/baseline/2025-04-11-11-14-04"               # example output path

strategy="ekfac"

python fit_factors.py \
    --model_name $model \
    --factor_strategy $strategy \
    --checkpoint_dir $ckpt_dir \
    --output_dir $output_dir \
    --train_batch_size 4 \
    --use_half_precision \