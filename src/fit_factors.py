"""
Script to fit influence factors (inverse Hessian approximation) for a given model and dataset.
"""
import argparse
import logging
import os
import numpy as np

import torch
from transformers import default_data_collator
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

from utils.datasets import get_tokenized_openwebtext
from utils.models import construct_hf_model, get_registry_config
from utils.tasks import LanguageModelingTask


def parse_args():
    parser = argparse.ArgumentParser(description="Fit influence factors (inverse Hessian approximation).")

    # model and dataset
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
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the output.",
    )

    # IF arguments
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
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
        "--covariance_module_partitions",
        type=int,
        default=1,
        help="Number of partitions for computing the covariance matrix across model modules.",
    )
    parser.add_argument(
        "--lambda_module_partitions",
        type=int,
        default=1,
        help="Number of partitions for computing the lambda matrix across model modules.",
    )
    parser.add_argument(
        "--covariance_data_partitions",
        type=int,
        default=1,
        help="Number of partitions for computing the covariance matrix across data points.",
    )
    parser.add_argument(
        "--lambda_data_partitions",
        type=int,
        default=1,
        help="Number of partitions for computing the lambda matrix across data points.",
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

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load config
    config = get_registry_config(args.model_name)

    # Prepare the trained model and dataset
    model = construct_hf_model(config, checkpoint_dir=args.checkpoint_dir)

    train_indices = np.load(args.train_indices_path)
    train_dataset = get_tokenized_openwebtext(config, train_indices)

    # Setup IF task
    task = LanguageModelingTask(config)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    if args.use_compile:
        model = torch.compile(model)

    if args.output_dir is None:
        args.output_dir = os.path.join("..", args.checkpoint_dir) if args.checkpoint_dir else f"models/openwebtext/{args.model_name}"
    
    analyzer = Analyzer(
        analysis_name="if_results",
        model=model,
        task=task,
        profile=args.profile,
        output_dir=args.output_dir,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    if args.use_compile:
        factors_name += "_compile"
    
    factor_args.covariance_module_partitions = args.covariance_module_partitions
    factor_args.lambda_module_partitions = args.lambda_module_partitions
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions

    factor_args.covariance_max_examples = None  # No limit
    factor_args.lambda_max_examples = None

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )


if __name__ == "__main__":
    main()
