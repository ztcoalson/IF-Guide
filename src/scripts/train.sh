#!/bin/bash

source scripts/helper_scripts/find_open_port.sh
port=$(find_open_port)
echo "Using port $port"

num_gpus=1

model="pythia-160m"
save_id="baseline"

torchrun --nproc-per-node=$num_gpus --master_port=$port train.py \
    --model_name $model \
    --learning_rate 6e-4 \
    --weight_decay 4e-4 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_train_epochs 4 \
    --logging_steps 25 \
    --seed 1004 \
    --save_id $save_id \
    # --toxic_token_mask_path "" \
    # --toxic_lambda 1.0 \