"""
Script to finetune a model on the OpenWebText dataset
with support for suppressing toxic tokens identified by IF-Guide.
"""
import os
import time
import json
import torch
import logging
import numpy as np
import argparse

from transformers import Trainer, TrainingArguments

from utils.datasets import get_tokenized_openwebtext
from utils.models import construct_hf_model, get_registry_config
from utils.training import SuppressToxicTokensTrainer, LogToFileCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model on OpenWebText dataset.")

    # Model and datasets
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF Model name to train (see utils/registry.yaml).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the model checkpoint to load.",
    )

    # Output/saving
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Directory for saving models.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--save_id",
        type=str,
        default="",
        help="ID for saving.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Checkpoint to resume from (if provided, else fresh finetune).",
    )

    # Data
    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--train_indices_path",
        type=str,
        default="../data/openwebtext/train_indices.npy",
        help="Path to the training indices.",
    )
    parser.add_argument(
        "--valid_indices_path",
        type=str,
        default="../data/openwebtext/valid_indices.npy",
        help="Path to the validation indices.",
    )

    # Training
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-05,
        help="Starting learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=4e-4,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.01,
        help="Ratio of total training steps to perform learning rate warmup.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for model training.",
    )
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=256,
        help="Effective batch size for training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Total number of steps to train the model.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=False,
        help="Whether to compile the model using torch.compile.",
    )

    # IF-Guide
    parser.add_argument(
        "--toxic_token_mask_path",
        type=str,
        default=None,
        help="Path to the toxic token mask to apply to the training dataset.",
    )
    parser.add_argument(
        "--toxic_lambda",
        type=float,
        default=1.0,
        help="Factor to multiply the negation mask by.",
    )

    return parser.parse_args() 


if __name__ == "__main__":
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.save_dir = os.path.join(
        args.save_dir, 
        "finetuning", 
        args.model_name, 
        args.save_id, 
        time.strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    config = get_registry_config(args.model_name)

    # Build model
    model = construct_hf_model(registry_config=config, checkpoint_dir=args.checkpoint_dir)
    if args.compile_model:
        model = torch.compile(model)

    # Build train (finetuning) dataset
    train_indices = np.load(args.train_indices_path).tolist()
    valid_indices = np.load(args.valid_indices_path).tolist()
    train_dataset = get_tokenized_openwebtext(config, indices=train_indices)
    eval_dataset = get_tokenized_openwebtext(config, indices=valid_indices)
    
    # Attach toxic token masks if provided
    if args.toxic_token_mask_path is not None:
        logging.info(f"Loading toxic token mask from {args.toxic_token_mask_path}")
        toxic_token_mask = torch.load(args.toxic_token_mask_path, weights_only=True)
        train_dataset = train_dataset.map(
            lambda example, idx: {"toxic_token_mask": toxic_token_mask[idx].tolist()},
            with_indices=True,
            desc="Attaching toxic token masks",
            num_proc=4
        )
        eval_dataset = eval_dataset.map(
            lambda example, idx: {"toxic_token_mask": [False] * len(example["input_ids"])},
            with_indices=True,
            desc="Attaching toxic token masks",
            num_proc=4
        )

    logging.info(f"Finetuning dataset size: {len(train_dataset)} ({len(train_dataset) * config['model']['block_size'] / 1e9:.2f} billion tokens)")
    logging.info(f"Validation dataset size: {len(eval_dataset)} ({len(eval_dataset) * config['model']['block_size'] / 1e9:.2f} billion tokens)")
    
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = max(1, args.effective_batch_size // (args.train_batch_size * num_gpus))
    logging.info(f"Using {num_gpus} GPU(s). Setting gradient_accumulation_steps to {gradient_accumulation_steps} for effective batch size {args.effective_batch_size}.")

    # Train model
    training_args = TrainingArguments(
        lr_scheduler_type="cosine",
        output_dir=args.save_dir,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=args.logging_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-08,
        remove_unused_columns=False,
        bf16=True,
        save_strategy="steps",
        save_steps=250,
        ddp_find_unused_parameters=False,
    )
    
    if args.toxic_token_mask_path is not None:
        trainer = SuppressToxicTokensTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[LogToFileCallback(os.path.join(args.save_dir, "log.out"))],
            toxic_lambda=args.toxic_lambda,
        )
    else:
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[LogToFileCallback(os.path.join(args.save_dir, "log.out"))],
        )
    # trainer.can_return_loss = True  # uncomment if using LoRA: https://github.com/huggingface/peft/issues/1881

    trainer.train(resume_from_checkpoint=args.resume_path)