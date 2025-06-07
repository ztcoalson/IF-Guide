#!/bin/bash

model="pythia-160m"
scores_path="models/training/pythia-160m/baseline/2025-04-11-11-14-04/if_results/scores_RTP/ekfac_half/pairwise_scores.safetensors" # example scores path

python build_toxic_token_mask.py \
    --model_name $model \
    --window 1 \
    --toxicity_threshold 0.99 \
    --scores_path $scores_path \
    --inspection_idx 1 \