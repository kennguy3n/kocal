"""Dataset classes for kchat-encoder multi-task training.

Provides PyTorch Dataset implementations for:
- SafetyDataset: 10-class safety classification from eval JSON datasets
- EmbeddingDataset: Contrastive query-document pairs for embedding
- RerankDataset: Query-document pairs with relevance labels for reranking
- MultiTaskDataset: Combined dataset that samples from all three task datasets
"""

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(SAFETY_CATEGORIES)}
MAX_SEQ_LEN = 512


class SafetyDataset(Dataset):
    """Safety classification dataset from eval JSON."""

    def __init__(self, json_path: str, tokenizer: Any):
        self.tokenizer = tokenizer
        self.samples: list[dict] = []
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            text = item.get("text", "")
            category = item.get("category", "SAFE")
            label = CATEGORY_TO_IDX.get(category, 0)
            self.samples.append({"text": text, "label": label})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "safety_label": torch.tensor(sample["label"], dtype=torch.long),
            "task": "safety",
        }


class EmbeddingDataset(Dataset):
    """Contrastive embedding dataset with query-document pairs."""

    def __init__(self, pairs: list[dict], tokenizer: Any):
        self.tokenizer = tokenizer
        self.samples = pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "task": "embed",
        }


class RerankDataset(Dataset):
    """Cross-encoder reranking dataset with query-document pairs."""

    def __init__(self, pairs: list[dict], tokenizer: Any):
        self.tokenizer = tokenizer
        self.samples = pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        query = sample["query"]
        doc = sample["document"]
        label = sample.get("label", 0)
        text = f"{query} [SEP] {doc}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rerank_label": torch.tensor(label, dtype=torch.float),
            "task": "rerank",
        }


class MultiTaskDataset(Dataset):
    """Combined multi-task dataset that samples from safety, embedding, and rerank."""

    def __init__(
        self,
        safety_dataset: SafetyDataset,
        embed_dataset: EmbeddingDataset | None = None,
        rerank_dataset: RerankDataset | None = None,
        safety_weight: float = 0.5,
        embed_weight: float = 0.25,
        rerank_weight: float = 0.25,
    ):
        self.safety = safety_dataset
        self.embed = embed_dataset
        self.rerank = rerank_dataset
        self.weights = [safety_weight, embed_weight, rerank_weight]

        total_len = len(self.safety)
        if self.embed:
            total_len = max(total_len, len(self.embed))
        if self.rerank:
            total_len = max(total_len, len(self.rerank))
        self._len = total_len

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tasks = ["safety"]
        if self.embed:
            tasks.append("embed")
        if self.rerank:
            tasks.append("rerank")

        task = random.choices(tasks, weights=self.weights[: len(tasks)])[0]

        if task == "safety":
            return self.safety[idx % len(self.safety)]
        elif task == "embed":
            return self.embed[idx % len(self.embed)]
        else:
            return self.rerank[idx % len(self.rerank)]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for multi-task batches."""
    task = batch[0]["task"]
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "task": task,
    }
    if task == "safety":
        result["labels"] = torch.stack([b["safety_label"] for b in batch])
    elif task == "rerank":
        result["labels"] = torch.stack([b["rerank_label"] for b in batch])
    return result
