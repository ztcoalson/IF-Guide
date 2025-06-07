"""
External toxicity classifiers.
Detoxify is a package, so we just include ToxiGen-RoBERTa here.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "tomh/toxigen_roberta"

class ToxigenRoBERTa:
    def __init__(self, device):
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = device
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[int]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        return probs[:, 1].tolist()