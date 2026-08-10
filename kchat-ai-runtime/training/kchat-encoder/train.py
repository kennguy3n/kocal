"""Multi-task training script for kchat-encoder.

Trains a unified XLM-RoBERTa-base model on safety classification,
text embedding, and cross-encoder reranking simultaneously.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import (
    SafetyDataset,
    EmbeddingDataset,
    RerankDataset,
    MultiTaskDataset,
    collate_fn,
)
from model import KchatEncoder


def train(
    base_model: str = "xlm-roberta-base",
    output_dir: str = "./checkpoints/",
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    safety_weight: float = 1.0,
    embed_weight: float = 0.5,
    rerank_weight: float = 0.5,
    data_dir: str = "./data/",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Device: {device}")
    print(f"Base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = KchatEncoder(base_model).to(device)

    safety_path = os.path.join(data_dir, "safety.json")
    embed_path = os.path.join(data_dir, "embed.json")
    rerank_path = os.path.join(data_dir, "rerank.json")

    print("Loading datasets...")
    safety_ds = SafetyDataset(safety_path, tokenizer)
    print(f"  Safety: {len(safety_ds)} samples")

    embed_ds = None
    if os.path.exists(embed_path):
        import json
        with open(embed_path, encoding="utf-8") as f:
            embed_data = json.load(f)
        embed_ds = EmbeddingDataset(embed_data, tokenizer)
        print(f"  Embedding: {len(embed_ds)} samples")

    rerank_ds = None
    if os.path.exists(rerank_path):
        import json
        with open(rerank_path, encoding="utf-8") as f:
            rerank_data = json.load(f)
        rerank_ds = RerankDataset(rerank_data, tokenizer)
        print(f"  Reranking: {len(rerank_ds)} samples")

    multi_ds = MultiTaskDataset(
        safety_ds, embed_ds, rerank_ds,
        safety_weight=safety_weight,
        embed_weight=embed_weight,
        rerank_weight=rerank_weight,
    )

    loader = DataLoader(
        multi_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    safety_loss_fn = nn.CrossEntropyLoss()
    rerank_loss_fn = nn.BCEWithLogitsLoss()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    step = 0

    print(f"\nTraining for {epochs} epochs ({total_steps} steps)...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task = batch["task"]

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, task=task)

            if task == "safety":
                labels = batch["labels"].to(device)
                loss = safety_loss_fn(outputs["safety_logits"], labels)
            elif task == "rerank":
                labels = batch["labels"].to(device).unsqueeze(-1)
                loss = rerank_loss_fn(outputs["rerank_score"], labels)
            elif task == "embed":
                embeddings = outputs["embedding"]
                if embeddings.size(0) > 1:
                    sim = embeddings @ embeddings.T
                    labels = torch.arange(embeddings.size(0), device=device)
                    loss = nn.functional.cross_entropy(sim, labels)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            if step % 100 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs} | Step {step}/{total_steps} | Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}\n")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = output_path / "best" / "model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(str(checkpoint_path))
            tokenizer.save_pretrained(str(output_path / "best"))
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

    final_path = output_path / "final" / "model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(final_path))
    tokenizer.save_pretrained(str(output_path / "final"))
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train kchat-encoder multi-task model")
    parser.add_argument("--base-model", default="xlm-roberta-base")
    parser.add_argument("--output-dir", default="./checkpoints/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--embed-weight", type=float, default=0.5)
    parser.add_argument("--rerank-weight", type=float, default=0.5)
    parser.add_argument("--data-dir", default="./data/")
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        safety_weight=args.safety_weight,
        embed_weight=args.embed_weight,
        rerank_weight=args.rerank_weight,
        data_dir=args.data_dir,
    )
