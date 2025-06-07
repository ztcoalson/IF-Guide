#!/bin/bash

source scripts/helper_scripts/find_open_port.sh
port=$(find_open_port)
echo "Using port $port"

num_gpus=1

model="pythia-160m"
save_id="baseline"
ckpt_dir="models/training/pythia-160m/baseline/2025-04-11-11-14-04/checkpoint-7628" # example checkpoint path

torchrun --nproc-per-node=$num_gpus --master_port=$port finetune.py \
    --model_name $model \
    --learning_rate 6e-5 \
    --weight_decay 4e-4 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_steps 2000 \
    --logging_steps 25 \
    --seed 1004 \
    --save_id $save_id \
    --checkpoint_dir $ckpt_dir \
    --toxic_token_mask_path "../data/toxic_token_masks/ekfac_RTP_pythia-160m/mask_toks=20.0m_p=0.99_w=1.pt" \
    --toxic_lambda 1.0 \