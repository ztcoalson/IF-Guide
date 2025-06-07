"""
Script to compute token-wise, differential influence scores for an LLM
on the provided training and query datasets.
"""
import argparse
import logging
import os
import numpy as np
from typing import Dict

import torch
from datetime import timedelta
from transformers import default_data_collator
from accelerate import Accelerator, InitProcessGroupKwargs

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

from utils.datasets import get_tokenized_openwebtext, get_tokenized_rtp
from utils.models import construct_hf_model, get_registry_config
from utils.tasks import LanguageModelingTask

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute influence scores.")

    # Model and datasets
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to be analyzed.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the target model checkpoint.",
    )
    parser.add_argument(
        "--train_indices_path",
        type=str,
        default="../data/openwebtext/train_indices.npy",
        help="Path to the training indices.",
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        choices=["RTP"],
        default="RTP",
        help="Which evaluation dataset to use. Options: 'RTP'.",
    )
    parser.add_argument(
        "--toxic_query_indices_path",
        type=str,
        help="Path to the toxic indices for the query dataset.",
    )
    parser.add_argument(
        "--nontoxic_query_indices_path",
        type=str,
        help="Path to the nontoxic indices for the query dataset.",
    )

    # Output/saving
    parser.add_argument(
        "--save_id",
        type=str,
        default=None,
        help="ID to append to the output names.",
    )
    parser.add_argument(    
        "--save_dir",
        type=str,
        default="",
        help="Directory to save the results.",
    )

    # IF arguments
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factors_name",
        type=str,
        default="ekfac_factors",
        help="Name of factors directory.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    parser.add_argument(
        "--factors_path",
        type=str,
        default=None,
        help="Path to the factors file.",
    )

    return parser.parse_args()


def get_query_dataset(config, dset_name, indices=None):
    if dset_name == "RTP":
        dset = get_tokenized_rtp(config, indices, prompts_only=False)
    else:
        raise ValueError(
            f"Unknown query_dataset '{dset_name}'. "
            "Please choose from ['RTP']."
        )
    
    return dset


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load model and token config
    config = get_registry_config(args.model_name)

    # Prepare the trained model
    model = construct_hf_model(config, args.checkpoint_dir)

    # Prepare the datasets
    train_dataset = get_tokenized_openwebtext(config, np.load(args.train_indices_path))

    toxic_query_dataset = get_query_dataset(config, args.query_dataset, np.load(args.toxic_query_indices_path))
    nontoxic_query_dataset = get_query_dataset(config, args.query_dataset, np.load(args.nontoxic_query_indices_path))

    # Define task and prepare model
    task = LanguageModelingTask(config)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    if args.use_compile:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name="if_results",
        model=model,
        task=task,
        profile=args.profile,
        output_dir=args.factors_path,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = args.factor_strategy

    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    if args.use_compile:
        scores_name += "_compile"
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"
    if args.save_id:
        scores_name += f"_{args.save_id}"

    # We always compute per-token scores and aggregate query gradients
    score_args.compute_per_token_scores = True
    score_args.aggregate_query_gradients = True

    if args.save_dir:
        scores_name = os.path.join(args.save_dir, scores_name)

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=toxic_query_dataset,
        negative_query_dataset=nontoxic_query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
