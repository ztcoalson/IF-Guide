#!/bin/bash

outputs_path="results/RTP/pythia-160m/baseline/outputs.json"
dataset="RTP"

python run_implicit_toxicity_eval.py \
    --outputs_file_path $outputs_path \
    --dataset $dataset \
    --batch_size 128