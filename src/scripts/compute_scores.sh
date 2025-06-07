#!/bin/bash

model="pythia-160m"
ckpt_dir="models/training/pythia-160m/baseline/2025-04-11-11-14-04/checkpoint-7628" # example checkpoint path
factors_path="models/training/pythia-160m/baseline/2025-04-11-11-14-04/"            # example factors path

dset_name="RTP"

python compute_scores.py \
    --model_name $model \
    --query_dataset $dset_name \
    --toxic_query_indices_path "../data/${dset_name}/query_indices/toxic_indices.npy" \
    --nontoxic_query_indices_path "../data/${dset_name}/query_indices/nontoxic_indices.npy" \
    --save_dir $dset_name \
    --factor_strategy "ekfac" \
    --factors_name "ekfac_half" \
    --checkpoint_dir $ckpt_dir \
    --factors_path $factors_path \
    --query_batch_size 4 \
    --train_batch_size 4 \
    --use_half_precision \