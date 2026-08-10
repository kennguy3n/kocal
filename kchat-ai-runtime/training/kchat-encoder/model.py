"""Multi-task model definition for kchat-encoder.

Shared XLM-RoBERTa-base encoder with three task heads:
- Safety classification (10 classes)
- Text embedding (768-dim, L2-normalized)
- Cross-encoder reranking (single relevance score)
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig

SAFETY_CATEGORIES = [
    "SAFE",
    "HARASSMENT",
    "HATE_SPEECH",
    "SELF_HARM",
    "VIOLENCE",
    "SEXUAL_CONTENT",
    "CHILD_SAFETY",
    "SCAM",
    "PII",
    "URL_RISK",
]

NUM_SAFETY_CLASSES = len(SAFETY_CATEGORIES)
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512


class KchatEncoder(nn.Module):
    """Multi-task XLM-RoBERTa-base encoder with safety, embedding, and rerank heads."""

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.safety_head = nn.Linear(hidden_size, NUM_SAFETY_CLASSES)
        self.embedding_head = nn.Linear(hidden_size, EMBEDDING_DIM)
        self.rerank_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str = "safety",
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        result: dict[str, torch.Tensor] = {}

        if task in ("safety", "all"):
            result["safety_logits"] = self.safety_head(pooled)

        if task in ("embed", "all"):
            emb = self.embedding_head(pooled)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            result["embedding"] = emb

        if task in ("rerank", "all"):
            score = self.rerank_head(pooled)
            result["rerank_score"] = score

        return result

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.encoder.config.to_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str) -> "KchatEncoder":
        checkpoint = torch.load(path, map_location="cpu")
        config = XLMRobertaConfig.from_dict(checkpoint["config"])
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.encoder = XLMRobertaModel(config)
        model.safety_head = nn.Linear(config.hidden_size, NUM_SAFETY_CLASSES)
        model.embedding_head = nn.Linear(config.hidden_size, EMBEDDING_DIM)
        model.rerank_head = nn.Linear(config.hidden_size, 1)
        model.load_state_dict(checkpoint["state_dict"])
        return model
