"""Export kchat-encoder PyTorch checkpoint to ONNX with optional quantization.

Exports the multi-task XLM-RoBERTa-base model to ONNX format with three
output heads: safety_logits, embedding, and rerank_score.
Supports INT8 (dynamic quantization) and INT4 (static quantization) export.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import KchatEncoder, NUM_SAFETY_CLASSES, EMBEDDING_DIM, MAX_SEQ_LEN


class OnnxKchatEncoder(nn.Module):
    """Wrapper for ONNX export that outputs all three heads simultaneously."""

    def __init__(self, model: KchatEncoder):
        super().__init__()
        self.encoder = model.encoder
        self.safety_head = model.safety_head
        self.embedding_head = model.embedding_head
        self.rerank_head = model.rerank_head

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled = last_hidden_state[:, 0, :]

        safety_logits = self.safety_head(pooled)
        embedding = torch.nn.functional.normalize(
            self.embedding_head(pooled), p=2, dim=-1
        )
        rerank_score = self.rerank_head(pooled)

        return safety_logits, embedding, rerank_score, last_hidden_state


def export_onnx(
    checkpoint_dir: str,
    output_dir: str,
    quantize: str | None = None,
    opset: int = 17,
):
    print(f"Loading checkpoint from {checkpoint_dir}...")
    checkpoint_path = Path(checkpoint_dir) / "model.pt"
    model = KchatEncoder.from_checkpoint(str(checkpoint_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    onnx_model = OnnxKchatEncoder(model)
    onnx_model.eval()

    dummy_input = tokenizer(
        "This is a test sentence for ONNX export.",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_file = output_path / "kchat-encoder.onnx"

    print(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        onnx_model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_file),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["safety_logits", "embedding", "rerank_score", "last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "safety_logits": {0: "batch_size"},
            "embedding": {0: "batch_size"},
            "rerank_score": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"  Saved: {onnx_file}")

    tokenizer.save_pretrained(output_path / "tokenizer")

    if quantize == "int8":
        print("Applying INT8 dynamic quantization...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantized_file = output_path / "kchat-encoder-int8.onnx"
            quantize_dynamic(
                str(onnx_file),
                str(quantized_file),
                weight_type=QuantType.QUInt8,
            )
            print(f"  Saved: {quantized_file}")
        except ImportError:
            print("  onnxruntime not available, skipping INT8 quantization")

    elif quantize == "int4":
        print("Applying INT4 static quantization...")
        try:
            from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
            import numpy as np

            class DummyCalibrationReader(CalibrationDataReader):
                def __init__(self, tokenizer, max_samples=100):
                    self.tokenizer = tokenizer
                    self.samples = [
                        "Hello world",
                        "This is a test message",
                        "Xin chào thế giới",
                        "你好世界",
                        "こんにちは",
                    ]
                    self.idx = 0
                    self.max_samples = max_samples

                def get_next(self):
                    if self.idx >= min(len(self.samples), self.max_samples):
                        return None
                    text = self.samples[self.idx]
                    self.idx += 1
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                        padding="max_length",
                        return_tensors="np",
                    )
                    return {
                        "input_ids": encoding["input_ids"].astype(np.int64),
                        "attention_mask": encoding["attention_mask"].astype(np.int64),
                    }

            calibration_data = output_path / "calibration"
            calibration_data.mkdir(exist_ok=True)

            reader = DummyCalibrationReader(tokenizer)
            quantized_file = output_path / "kchat-encoder-int4.onnx"

            quantize_static(
                str(onnx_file),
                str(quantized_file),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt4,
                per_channel=True,
                reduce_range=True,
            )
            print(f"  Saved: {quantized_file}")
        except ImportError:
            print("  onnxruntime not available, skipping INT4 quantization")

    print("\nExport complete.")
    print(f"  Base model: {onnx_file}")
    if quantize:
        print(f"  Quantized: {quantize}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export kchat-encoder to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best/",
        help="Path to checkpoint directory (containing model.pt and tokenizer files)",
    )
    parser.add_argument(
        "--output-dir",
        default="./onnx/",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "int4", "none"],
        default="none",
        help="Quantization mode (int8, int4, or none)",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        quantize=args.quantize if args.quantize != "none" else None,
        opset=args.opset,
    )
