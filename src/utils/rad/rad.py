"""
Implementation of the RAD decoding-time defense.
Credit: https://github.com/r-three/RAD
"""
import torch

from transformers import (
    LogitsProcessorList,
    TopPLogitsWarper,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
)
from .utils.logits_processor import (
    RewardAugmentedLogitsProcessor,
    RewardAugmentedLogitsProcessorNoPkv
)
from .reward_modeling.reward_model import GPT2RewardModel

RM_DIR = "utils/rad/reward_modeling/saved_models/gpt2_toxicity/pytorch_model.bin"

class RewardAugmentedDecoder():
    
    def __init__(self, language_model, lm_tokenizer, reward_model, rm_tokenizer, 
                 max_length, num_gpus=1, inverse=False, efficient=True):
        self._lm = language_model
        self._lm_tokenizer = lm_tokenizer
        self._rm = reward_model
        self._rm_tokenizer = rm_tokenizer
        self._max_length = max_length
        self._num_gpus = num_gpus
        self._inverse = inverse
        self._efficient = efficient
        self.device = self._lm.device
        self._vocab_size = len(self._lm_tokenizer)

    def generate(
        self, 
        input_ids,
        attention_mask,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=20,
        return_dict_in_generate=True,
        output_scores=False,
        pad_token_id=None,
        beta=50,
        data_container=None,
        method="linear",
        top_k=None,     # dummy
    ):
        if self._efficient:
            logits_processor = LogitsProcessorList([
                TopPLogitsWarper(top_p=top_p),
                RewardAugmentedLogitsProcessor(
                    self._lm_tokenizer,
                    self._rm_tokenizer,
                    self._rm,
                    topk=20,     # consider all tokens
                    method=method,
                    beta=beta,
                    num_gpus=self._num_gpus,
                    inverse=self._inverse,
                    data_container=data_container
                ),
            ])
            
        else:
            logits_processor = LogitsProcessorList([
                TopPLogitsWarper(top_p=top_p),
                RewardAugmentedLogitsProcessorNoPkv(
                    self._lm_tokenizer,
                    self._rm_tokenizer,
                    self._rm,
                    topk=20,     # consider all tokens
                    method=method,
                    beta=beta,
                    inverse=self._inverse,
                ),
            ])
                
        outputs = self._lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            pad_token_id=pad_token_id,
        )
        
        return outputs


def load_rad(
    base_model: PreTrainedModel,
    base_tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    inverse: bool = True,
    rm: str = "gpt2",
) -> RewardAugmentedDecoder:
    """
    Load the RAD decoder with the given base model and tokenizer.

    Args:
        base_model (PreTrainedModel): The base language model.
        base_tokenizer (PreTrainedTokenizer): The tokenizer for the base model.
        max_length (int, optional): Maximum length for input sequences. Defaults to 2048.

    Returns:
        RewardAugmentedDecoder: An instance of the RAD decoder.
    """
    # rm
    if rm == 'gpt2':
        rm_tokenizer = AutoTokenizer.from_pretrained(rm)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.padding_side = 'right'
        rm_tokenizer.max_length = 1024
        
        rm = GPT2RewardModel(reward_model_name=rm, out_features=7)
        
        state_dict = torch.load(RM_DIR, weights_only=True)
        rm.load_state_dict(state_dict, strict=False)
        rm = rm.to("cuda")
    else:
        raise ValueError(f"Unknown reward model: {rm}")


    rad = RewardAugmentedDecoder(
        base_model, 
        base_tokenizer, 
        rm, 
        rm_tokenizer, 
        max_length, 
        num_gpus=torch.cuda.device_count(),
        inverse=inverse)
    return rad
