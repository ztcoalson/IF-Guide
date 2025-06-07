"""
Utilities for loading datasets.
"""
import copy
import torch
import logging

from typing import List, Dict
from itertools import chain
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

TOKENIZED_OWT_PATH = "ztcoalson/openwebtext-gptneox-2048-chunks"
REAL_TOXICITY_PROMPTS_PATH = "../data/RTP/rtp.json"     # RTP with toxicity classifications from Detoxify instead of Perspective API
ATTAQ_PATH = "ibm-research/AttaQ"
BOLD_PATH = "AmazonScience/bold"
LAMBADA_PATH = "EleutherAI/lambada_openai"


def build_tokenized_loader(
    tokenized_dataset: Dataset,
    batch_size: int = 16,
):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([torch.tensor(sample[key]) for sample in batch])
            for key in batch[0].keys()
        }

    loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    return loader


def get_tokenized_openwebtext(
    config: dict,
    indices: List[int] = None,
) -> Dataset:
    if config['tokenizer']['name'] != "pythia-tok" or config['model']['block_size'] != 2048:
        raise ValueError(
            "Only 'pythia-tok' tokenizer and 'block_size' = 2048 are supported for the OpenWebText subset we provide ('ztcoalson/openwebtext-gptneox-2048-chunks' on HuggingFace Hub)."
            " We provide a 'tokenize_openwebtext' function in 'utils/datasets.py' to tokenize OpenWebText with different configurations."
            " If necessary, modify 'get_tokenized_openwebtext' in 'utils/datasets.py' to load a different tokenized dataset using the aforementioned function."
        )

    dset = load_dataset(TOKENIZED_OWT_PATH, split="train")

    if indices is not None:
        dset = dset.select(indices)
    
    logging.info(f"Loaded tokenized OpenWebText from {TOKENIZED_OWT_PATH}.")
    
    return dset
        

def tokenize_openwebtext(
    tokenizer_path: str,
    block_size: int = 2048,
) -> Dataset:
    """
    Tokenizes the OpenWebText dataset into equal-sized chunks using the provided tokenizer.
    Adapted from https://github.com/pomonam/kronfluence/blob/main/examples/openwebtext/pipeline.py.

    Args:
        tokenizer_path (str): Path to the HuggingFace tokenizer.
        block_size (int): Size of the chunks to split the text into.
    """
    raw_dataset = load_dataset("openwebtext", split="train", num_proc=8)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)

    column_names = raw_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    ds = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return ds


def get_tokenized_rtp(
    config: dict,
    indices: List[int] = None,
    max_length: int = 256,      # max length in dataset is ~181 depending on tokenizer
    prompts_only: bool = False,
    dset_path: str = REAL_TOXICITY_PROMPTS_PATH,
) -> Dataset:
    if dset_path is None:
        dset_path = REAL_TOXICITY_PROMPTS_PATH
    data_kwargs = {
        "path": "json",
        "data_files": dset_path,
        "num_proc": 1,
    }
    raw_dataset = load_dataset(**data_kwargs)["train"]
    column_names = raw_dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['path'], 
        use_fast=True, 
        trust_remote_code=True,
        padding_side="left",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        if prompts_only:
            full_text = tokenizer(
                example['prompt']['text'], 
                truncation=True, 
                max_length=max_length, 
                padding="max_length", 
            )

            labels = copy.deepcopy(full_text["input_ids"])
            for j, mask in enumerate(full_text["attention_mask"]):
                if mask == 0:
                    labels[j] = -100
        else:    
            full_text = tokenizer(
                example['prompt']['text'] + example['continuation']['text'], 
                truncation=True, 
                max_length=max_length, 
                padding="max_length", 
            )

            # create labels
            prompt_length = len(tokenizer(example['prompt']['text'])["input_ids"])
            labels = copy.deepcopy(full_text["input_ids"])
            attention_mask = full_text["attention_mask"]
            first_non_pad_index = next(j for j, mask in enumerate(attention_mask) if mask == 1)

            # mask prompt tokens
            for j in range(first_non_pad_index, first_non_pad_index + prompt_length):
                labels[j] = -100 

            # mask all padding tokens
            for j, mask in enumerate(attention_mask):
                if mask == 0:
                    labels[j] = -100

        return {
            "index": example['index'],
            "input_ids": full_text["input_ids"],
            "attention_mask": full_text["attention_mask"],
            "labels": labels,
        }

    ds = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_tokenized_attaq(
    config: dict,
    indices: List[int] = None,
    max_length: int = 256,      # max length in dataset is ~101 depending on tokenizer
) -> Dataset:
    raw_dataset = load_dataset(ATTAQ_PATH, split="train", num_proc=4)
    raw_dataset = raw_dataset.add_column("index", list(range(len(raw_dataset))))
    column_names = raw_dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['path'], 
        use_fast=True, 
        trust_remote_code=True,
        padding_side="left",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        full_text = tokenizer(
            example['input'], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length", 
        )

        labels = copy.deepcopy(full_text["input_ids"])
        for j, mask in enumerate(full_text["attention_mask"]):
            if mask == 0:
                labels[j] = -100

        return {
            "index": example['index'],
            "input_ids": full_text["input_ids"],
            "attention_mask": full_text["attention_mask"],
            "labels": labels,
        }

    ds = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_tokenized_bold(
    config: dict,
    indices: List[int] = None,
    max_length: int = 128,      # max length in dataset is ~65 depending on tokenizer
) -> Dataset:
    raw_dataset = load_dataset(BOLD_PATH, split="train", num_proc=4)
    raw_dataset = raw_dataset.add_column("index", list(range(len(raw_dataset))))
    column_names = raw_dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['path'], 
        use_fast=True, 
        trust_remote_code=True,
        padding_side="left",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        full_text = tokenizer(
            example['prompts'][0], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length", 
        )

        labels = copy.deepcopy(full_text["input_ids"])
        for j, mask in enumerate(full_text["attention_mask"]):
            if mask == 0:
                labels[j] = -100

        return {
            "index": example['index'],
            "input_ids": full_text["input_ids"],
            "attention_mask": full_text["attention_mask"],
            "labels": labels,
        }

    ds = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_tokenized_lambada(
    config: dict,
    indices: List[int] = None,
    max_length: int = 512,      # max length in dataset is ~223 depending on tokenizer
) -> Dataset:
    raw_dataset = load_dataset(LAMBADA_PATH, "en", split="test", num_proc=1)
    raw_dataset = raw_dataset.add_column("index", list(range(len(raw_dataset))))
    column_names = raw_dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['path'], 
        use_fast=True, 
        trust_remote_code=True,
        padding_side="left",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(example):
        # Tokenize the full text
        full_text = tokenizer(
            example['text'],
            truncation=True, 
            max_length=max_length, 
            padding="max_length", 
        )

        # target is the last token
        labels = copy.deepcopy(full_text["input_ids"])

        return {
            "index": example['index'],
            "input_ids": full_text["input_ids"],
            "attention_mask": full_text["attention_mask"],
            "labels": labels,
        }

    ds = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if indices is not None:
        ds = ds.select(indices)

    return ds