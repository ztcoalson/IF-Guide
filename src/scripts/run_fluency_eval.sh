#!/bin/bash

model_name="pythia-160m"
ckpt_dir="models/training/pythia-160m/baseline/2025-04-11-11-14-04/checkpoint-7628" # example checkpoint path

defense="none"
save_dir="baseline"

python run_fluency_eval.py \
    --model_name $model_name \
    --checkpoint_dir $ckpt_dir \
    --batch_size 16 \
    --save_dir $save_dir \
    --decoding_defense $defense \