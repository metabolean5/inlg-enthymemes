"""Shared ROSCOE-SS (semantic similarity) encoder used by the three eval_*_roscoe.py scripts."""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROSCOE_MODEL_ID = "facebook/roscoe-512-roberta-base"


class RoscoeEncoder:
    def __init__(self, model_id=ROSCOE_MODEL_ID, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1)

    def encode(self, texts, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            embs = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs)


def roscoe_ss(a, b):
    """Cosine similarity rescaled to [0, 1]."""
    return float((1.0 + np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) / 2.0)
