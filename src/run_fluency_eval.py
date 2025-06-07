"""
Script for evaluating the fluency of an LLM, measured by:
1. Perplexity on OpenWebText
2. Last token accuracy on LAMBADA
"""
import os
import json
import math
import argparse
import torch
import numpy as np
import logging

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import set_seed, PreTrainedModel, AutoTokenizer

from utils.rad.rad import load_rad
from utils.models import construct_hf_model, get_registry_config
from utils.datasets import (
    build_tokenized_loader, 
    get_tokenized_openwebtext,
    get_tokenized_lambada
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fluency evaluation.")
    
    # Model and datasets
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for evaluation.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint of the model to evaluate. If none, loads the pretrained model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--owt_test_indicies",
        type=str,
        default="../data/openwebtext/test_indices.npy",
        help="Path to the OpenWebText test indicies.",
    )
    parser.add_argument(
        "--lambada_test_indicies",
        type=str,
        default="../data/lambada/test_indices.npy",
        help="Path to the LAMBADA test indicies.",
    )

    # Output/saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the evaluation results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="ID for saving the evaluation results.",
    )

    # General
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )

    # Decoding-time defenses
    parser.add_argument(
        "--decoding_defense",
        type=str,
        choices=['none', 'rad'],
        default='none',
        help="Decoding-time defense to use for evaluation.",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_last_token_accuracy(loader: DataLoader, model: PreTrainedModel, decoding_defense='none', pad_token_id=None):
    """
    Evaluate the model on the given dataset.
    """
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Evaluating"):
        labels = batch["labels"].to(model.device)

        last_token_labels = labels[:, -1]
        if decoding_defense == 'none':
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            last_token_logits = logits[:, -2, :]
        elif decoding_defense == 'rad':
            input_ids = batch["input_ids"][:, :-1].to(model.device)
            attention_mask = batch["attention_mask"][:, :-1].to(model.device)
            last_token_logits = model.generate(
                input_ids,
                attention_mask,
                do_sample=True,
                top_p=1.0,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_token_id,
            ).scores[0]

        # Get the predicted token
        predicted_tokens = torch.argmax(last_token_logits, dim=-1)
        probs = torch.softmax(last_token_logits, dim=-1)
        
        # Calculate accuracy
        correct += (predicted_tokens == last_token_labels).sum().item()
        total += last_token_labels.size(0)

    accuracy = correct / total

    return accuracy


@torch.no_grad()
def evaluate_perplexity(loader: DataLoader, model: PreTrainedModel):
    """
    Evaluate the model on the given dataset.
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        logits = model(input_ids, attention_mask=attention_mask).logits

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        
        # calculate loss
        loss = loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        ).view(shifted_labels.shape)

        total_loss += loss.sum().item()
        total_tokens += (shifted_labels != -100).sum().item()

    perplexity = math.exp(total_loss / total_tokens)   

    return perplexity


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Evaluating {args.model_name} on the fluency tasks.")
    
    torch.set_grad_enabled(False)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = os.path.join(
        args.output_dir, 
        "fluency",
        args.model_name,
        args.save_dir,
    )

    if os.path.exists(os.path.join(save_dir, "results.json")):
        with open(os.path.join(save_dir, "results.json"), "r") as f:
            current_results = json.load(f)
        with open(os.path.join(save_dir, "args.json"), "r") as f:
            current_args = json.load(f)

        assert current_args == vars(args), "Arguments have changed. Please delete the existing results or align the arguments."
    else:
        current_results = defaultdict(dict)
        current_args = {}

        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    logging.info(f"Save directory set to {save_dir}")

    # Load model config
    config = get_registry_config(args.model_name)
    
    # Set up model and HF text generation pipeline
    model = construct_hf_model(config, args.checkpoint_dir)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.decoding_defense == 'rad':
        model = load_rad(base_model=model, base_tokenizer=tokenizer)
        
    # Evaluate OpenWebText
    if current_results.get("openwebtext", {}).get("perplexity") is not None:
        logging.info("OpenWebText perplexity already computed. Skipping...")
    elif args.decoding_defense == 'rad':
        logging.info("RAD is incompatible with the OpenWebText evaluation. Skipping...")
    else:
        logging.info("Computing perplexity on OpenWebText...")
        owt_test_indicies = np.load(args.owt_test_indicies)
        owt_dataset = get_tokenized_openwebtext(config, owt_test_indicies)
        owt_dataset = owt_dataset.map(lambda example, idx: {"index": idx}, with_indices=True)
        owt_loader = build_tokenized_loader(owt_dataset, args.batch_size)
        
        owt_perplexity = evaluate_perplexity(owt_loader, model)
        logging.info(f"OpenWebText Perplexity: {owt_perplexity}")
        current_results["openwebtext"] = {
            "perplexity": owt_perplexity
        }
    
    # Evaluate LAMBADA
    if current_results.get("lambada", {}).get("last_token_accuracy") is not None:
        logging.info("LAMBADA last token accuracy already computed. Skipping...")
    else:
        logging.info("Computing last token accuracy on LAMBADA...")
        lambada_test_indicies = np.load(args.lambada_test_indicies)
        lambada_dataset = get_tokenized_lambada(config, lambada_test_indicies)
        lambada_dataset = lambada_dataset.map(lambda example, idx: {"index": idx}, with_indices=True)
        lambada_loader = build_tokenized_loader(lambada_dataset, args.batch_size)
        
        lambada_accuracy = evaluate_last_token_accuracy(
            lambada_loader, 
            model, 
            args.decoding_defense, 
            tokenizer.pad_token_id
        )
        logging.info(f"LAMBADA Last Token Accuracy: {lambada_accuracy}")
        current_results["lambada"] = {
            "last_token_accuracy": lambada_accuracy
        }

    # Save results
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(current_results, f, indent=4)


if __name__ == "__main__":
    main()