# kchat-encoder Training Pipeline

Training pipeline for the unified multi-task XLM-RoBERTa-base encoder model.

## Overview

The kchat-encoder is a single XLM-RoBERTa-base model fine-tuned for three tasks:

1. **Safety classification** (10 classes): SAFE, HARASSMENT, HATE_SPEECH, SELF_HARM,
   VIOLENCE, SEXUAL_CONTENT, CHILD_SAFETY, SCAM, PII, URL_RISK
2. **Text embedding** (768-dim): L2-normalized dense embeddings for semantic retrieval
3. **Cross-encoder reranking**: Relevance scoring for query-document pairs

## Architecture

```
XLM-RoBERTa-base (shared encoder)
├── [CLS] pooled output
│   ├── Safety head: Linear(768, 10) + softmax → 10-class classification
│   ├── Embedding head: Linear(768, 768) + L2 normalize → 768-dim embedding
│   └── Rerank head: Linear(768, 1) + sigmoid → relevance score
```

## Training Data

### Safety Classification
- **Source**: `eval/kchat-task-suite/datasets/safety/safety_dataset_v2.json`
- **Labels**: 10 categories (SAFE, HARASSMENT, HATE_SPEECH, SELF_HARM, VIOLENCE,
  SEXUAL_CONTENT, CHILD_SAFETY, SCAM, PII, URL_RISK)
- **Languages**: en, vi, zh, ja, ko, es, ar, de, hi, fr + mixed-language
- **Size**: ~2005 cases

### Text Embedding
- **Source**: MS-MARCO, mMARCO (multilingual), synthetic query-document pairs
- **Method**: Contrastive learning with in-batch negatives
- **Embedding dim**: 768

### Cross-Encoder Reranking
- **Source**: MS-MARCO passage ranking, mMARCO
- **Method**: Pointwise relevance scoring (binary cross-entropy)
- **Input**: Query-document pairs

## Usage

### 1. Prepare data

```bash
python prepare_data.py --output ./data/
```

### 2. Train (multi-task)

```bash
python train.py \
  --base-model xlm-roberta-base \
  --output-dir ./checkpoints/ \
  --batch-size 32 \
  --epochs 10 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --safety-weight 1.0 \
  --embed-weight 0.5 \
  --rerank-weight 0.5
```

### 3. Export to ONNX

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/best/ \
  --output-dir ./onnx/ \
  --quantize int8 \
  --opset 17
```

### 4. Quantize to INT4 (optional, for low-tier devices)

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/best/ \
  --output-dir ./onnx/ \
  --quantize int4 \
  --opset 17
```

## ONNX Model Outputs

The exported ONNX model has three output heads:

| Output Name | Shape | Description |
|-------------|-------|-------------|
| `safety_logits` | `[1, 10]` | Raw logits for 10 safety categories |
| `embedding` | `[1, 768]` | L2-normalized embedding vector |
| `rerank_score` | `[1, 1]` | Sigmoid relevance score for query-doc pair |

## Requirements

```
torch>=2.0
transformers>=4.36
onnx>=1.14
onnxruntime>=1.16
datasets>=2.14
accelerate>=0.24
```

## Files

| File | Purpose |
|------|---------|
| `prepare_data.py` | Download and format training data from eval datasets |
| `train.py` | Multi-task training script (safety + embedding + reranking) |
| `export_onnx.py` | Export PyTorch checkpoint to ONNX with optional quantization |
| `model.py` | Multi-task model definition (shared encoder + 3 heads) |
| `data.py` | Dataset classes for safety, embedding, and reranking tasks |
