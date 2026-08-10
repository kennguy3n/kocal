"""Prepare training data for kchat-encoder multi-task training.

Loads safety classification data from eval datasets and generates
synthetic embedding/reranking pairs from context eval data.
"""

import argparse
import json
from pathlib import Path

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


def load_safety_dataset(eval_path: str) -> list[dict]:
    """Load safety classification data from eval JSON."""
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        text = item.get("text", "")
        category = item.get("category", "SAFE")
        if category not in SAFETY_CATEGORIES:
            category = "SAFE"
        samples.append({"text": text, "category": category})
    return samples


def load_context_pairs(eval_path: str) -> tuple[list[dict], list[dict]]:
    """Load context eval data to create embedding and reranking pairs."""
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)

    embed_pairs = []
    rerank_pairs = []

    documents = data.get("documents", [])
    queries = data.get("queries", [])

    for doc in documents:
        embed_pairs.append({"text": doc.get("text", doc.get("content", ""))})

    for query in queries:
        q_text = query.get("query", query.get("text", ""))
        relevant = query.get("relevant_docs", [])
        for doc in documents:
            doc_id = doc.get("id", "")
            doc_text = doc.get("text", doc.get("content", ""))
            label = 1 if doc_id in relevant else 0
            rerank_pairs.append({
                "query": q_text,
                "document": doc_text,
                "label": label,
            })

    return embed_pairs, rerank_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare kchat-encoder training data")
    parser.add_argument(
        "--safety-json",
        default="../../eval/kchat-task-suite/datasets/safety/safety_dataset_v2.json",
        help="Path to safety eval JSON dataset",
    )
    parser.add_argument(
        "--context-json",
        default="../../eval/kchat-task-suite/datasets/context/context_eval.json",
        help="Path to context eval JSON dataset",
    )
    parser.add_argument(
        "--output",
        default="./data/",
        help="Output directory for prepared data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading safety dataset...")
    safety_samples = load_safety_dataset(args.safety_json)
    print(f"  {len(safety_samples)} safety samples")

    print("Loading context data for embedding/reranking pairs...")
    embed_pairs, rerank_pairs = [], []
    try:
        embed_pairs, rerank_pairs = load_context_pairs(args.context_json)
        print(f"  {len(embed_pairs)} embedding samples")
        print(f"  {len(rerank_pairs)} reranking pairs")
    except FileNotFoundError:
        print("  Context dataset not found, skipping embedding/reranking data")

    print("Writing prepared data...")
    with open(output_dir / "safety.json", "w", encoding="utf-8") as f:
        json.dump(safety_samples, f, ensure_ascii=False, indent=2)

    if embed_pairs:
        with open(output_dir / "embed.json", "w", encoding="utf-8") as f:
            json.dump(embed_pairs, f, ensure_ascii=False, indent=2)

    if rerank_pairs:
        with open(output_dir / "rerank.json", "w", encoding="utf-8") as f:
            json.dump(rerank_pairs, f, ensure_ascii=False, indent=2)

    print(f"Done. Data written to {output_dir}")


if __name__ == "__main__":
    main()
