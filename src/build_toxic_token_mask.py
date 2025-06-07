"""
Script to construct the toxic token masks used by IF-Guide to 
suppress toxicity during training. Uses token-wise influence scores
computed with `compute_scores.py`.
"""
import logging
import math
import os
import numpy as np
import argparse

import torch
from kronfluence.analyzer import Analyzer
from transformers import AutoTokenizer
from colorama import Fore, init
from tqdm import tqdm

from utils.models import get_registry_config
from utils.datasets import get_tokenized_openwebtext

init()


def parse_args():
    parser = argparse.ArgumentParser(description="Build toxic token mask using token-wise influence scores.")

    # Model and datasets
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of HF model to evaluate.",
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        default="RTP",
        choices=["RTP"],
        help="Dataset used to compute the query scores."
    )
    parser.add_argument(
        "--train_indices_path",
        type=str,
        default="../data/openwebtext/train_indices.npy",
        help="Path to the training indices for the dataset.",
    )

    # Output/saving
    parser.add_argument(
        "--save_id",
        type=str,
        default="",
        help="Identifier for the save path."
    )
    parser.add_argument(
        "--inspection_idx",
        type=int,
        default=0,
        help="Index of the example to inspect."
    )

    # IF arguments
    parser.add_argument(
        "--scores_path",
        type=str,
        default=None,
        help="Path to the influence scores.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        choices=["ekfac", "identity"],
        help="Method used for computing Hessian approximation."
    )

    # Arguments for our toxic token selection algorithm
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Window size for the mask."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20_000_000,
        help="Max number of tokens to penalize."
    )
    parser.add_argument(
        "--toxicity_threshold",
        type=float,
        default=0.99,
        help="Toxicity threshold for selecting toxic tokens."
    )

    return parser.parse_args()


def color_strength(word: str, strength: int) -> None:
    strength = max(0, min(1, strength))
    intensity = math.floor(strength * 255)
    color = f"\033[38;2;{intensity};0;0m"
    print(f"{color}{word}{Fore.RESET}", end="")


def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the harmonic mean of two tensors.
    """
    a = a.float()
    b = b.float()
    return 2 * a * b / (a + b + eps)


def build_toxic_token_mask(
    influence_scores: torch.Tensor,  # shape (num_samples, seq_len)
    toxicity_threshold: float = 0.99,
    context_window: int = 5,
    max_tokens: int = 1_000_000,
) -> torch.Tensor:
    """
    Builds a toxic token mask where:
      True = toxic token,
      False = non-toxic token.
    
    Ranking is done using the harmonic mean of:
      (1) the number of values above the toxicity threshold
      (2) the sum of those values
    """
    scores = influence_scores.detach()
    B, T = scores.shape

    # Compute threshold
    threshold = np.quantile(scores.cpu().float().numpy().flatten(), toxicity_threshold)

    # Compute stats per record
    above_thresh = (scores > threshold).float()
    count_above = min_max_normalize(above_thresh.sum(dim=1))         
    sum_above = min_max_normalize((scores * above_thresh).sum(dim=1))

    # Harmonic mean ranking
    harmonic_scores = harmonic_mean(count_above, sum_above)

    # Select top-k rows
    sorted_indices = torch.argsort(harmonic_scores, descending=True).tolist()

    mask = torch.zeros_like(scores, dtype=torch.bool)
    already_selected = torch.zeros_like(scores, dtype=torch.bool)
    used_tokens = 0

    pbar = tqdm(total=max_tokens, desc="Building toxic token mask", unit="tokens")
    for i in sorted_indices:
        toxic_indices = (scores[i] > threshold).nonzero(as_tuple=True)[0]
        for idx in toxic_indices:
            center = idx.item()
            start = max(0, center - context_window)
            end = min(T, center + context_window + 1)

            # Get the span that hasn't been selected yet
            new_mask = ~already_selected[i, start:end]
            new_tokens = new_mask.sum().item()

            if used_tokens + new_tokens > max_tokens:
                print(f"Reached max token limit of {max_tokens}. Stopping...")
                return mask, sorted_indices[:i]

            # Select only where needed
            mask[i, start:end][new_mask] = True
            already_selected[i, start:end][new_mask] = True

            used_tokens += new_tokens
            pbar.update(new_tokens)

    return mask, sorted_indices 


def main():
    args = parse_args()

    config = get_registry_config(args.model_name)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Building/loading toxic token mask for {args.model_name}...")

    # Prepare the dataset.
    train_dataset = get_tokenized_openwebtext(config, np.load(args.train_indices_path))
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'], use_fast=True, trust_remote_code=True)

    save_path = (
        f"../data/toxic_token_masks/"
        f"{args.factor_strategy}_{args.query_dataset}_{args.model_name}"
        f"{'_' + args.save_id if args.save_id else ''}/"
        f"mask_toks={round(args.max_tokens / 1e6, 1)}m"
        f"_p={args.toxicity_threshold}"
        f"_w={args.window}.pt"
    )

    if os.path.exists(save_path):
        mask = torch.load(save_path, map_location="cpu", weights_only=True)
        logging.info(f"Loaded mask from {save_path}")
    else:
        logging.info(f"Building toxic token mask and saving to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        scores = Analyzer.load_file(args.scores_path)['all_modules'][0]

        mask, _ = build_toxic_token_mask(scores, toxicity_threshold=args.toxicity_threshold, context_window=args.window, max_tokens=args.max_tokens)
        torch.save(mask, save_path)
    
    top_indices = torch.argsort((mask == True).sum(dim=1), descending=True).tolist()
    top_idx = top_indices[args.inspection_idx]

    # Get the corresponding sequences
    logging.info("Influential example (selected toxic tokens in red):")
    words = tokenizer.batch_decode(train_dataset[top_idx]["input_ids"])
    strengths = mask[top_idx] == True

    for word, strength in zip(words, strengths):
        color_strength(word, strength)

if __name__ == "__main__":
    main()
