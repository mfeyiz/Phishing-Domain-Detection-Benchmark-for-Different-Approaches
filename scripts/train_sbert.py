#!/usr/bin/env python3
"""
SBERT Bi-Encoder Training Script

Fine-tunes SBERT model for phishing detection using contrastive learning.
Training data: data/train.json
Validation: data/val.json

Usage:
    python scripts/train_sbert.py

Output:
    models/sbert_model/ - Fine-tuned SBERT model
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    evaluation,
)
from torch.utils.data import DataLoader

from src.config import get_settings
from src.detectors.sbert_detector import predict, evaluate

settings = get_settings()


def load_data(data_path: str):
    """Load training data from JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def create_training_examples(data: list):
    """Create InputExample objects for SBERT training."""
    examples = []
    for item in data:
        orig = item["original"].split(".")[0].lower()
        susp = item["suspicious"].split(".")[0].lower()
        label = item["label"]

        example = InputExample(texts=[orig, susp], label=1.0 if label == 1 else 0.0)
        examples.append(example)

    return examples


def create_evaluator(val_data: list, threshold: float = 0.5):
    """Create evaluator for validation."""
    sentences1 = []
    sentences2 = []
    scores = []

    for item in val_data:
        orig = item["original"].split(".")[0].lower()
        susp = item["suspicious"].split(".")[0].lower()
        label = item["label"]

        sentences1.append(orig)
        sentences2.append(susp)
        scores.append(1.0 if label == 1 else 0.0)

    return evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=scores,
        name="phishing-detection",
    )


def train(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    epochs: int = 3,
    batch_size: int = 32,
    warmup_steps: int = 100,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_path: str = None,
):
    """Train SBERT model."""

    if output_path is None:
        output_path = settings.SBERT_MODEL_PATH

    print("=" * 60)
    print("SBERT Bi-Encoder Training")
    print("=" * 60)

    print(f"\n[1] Loading data...")
    train_data = load_data(train_path)
    print(f"    Training samples: {len(train_data)}")

    if val_path and os.path.exists(val_path):
        val_data = load_data(val_path)
        print(f"    Validation samples: {len(val_data)}")
    else:
        val_data = None
        print("    Validation: None (using train split)")

    print(f"\n[2] Creating training examples...")
    train_examples = create_training_examples(train_data)
    print(f"    Training examples: {len(train_examples)}")

    print(f"\n[3] Initializing model...")
    base_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        base_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[base_model, pooling_model])

    if torch.cuda.is_available():
        print("    Device: CUDA")
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        print("    Device: MPS (Apple Silicon)")
        model = model.to("mps")
    else:
        print("    Device: CPU")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    train_loss = losses.ContrastiveLoss(model=model)

    print(f"\n[4] Starting training...")
    print(f"    Epochs: {epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Warmup steps: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        evaluator=create_evaluator(val_data) if val_data else None,
        evaluation_steps=500,
    )

    print(f"\n[5] Model saved to: {output_path}")

    print(f"\n[6] Evaluating on test set...")
    test_results = evaluate("data/test.json")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return model, test_results


def main():
    parser = argparse.ArgumentParser(
        description="Train SBERT model for phishing detection"
    )
    parser.add_argument(
        "--train", type=str, default="data/train.json", help="Training data path"
    )
    parser.add_argument(
        "--val", type=str, default="data/val.json", help="Validation data path"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name",
    )
    parser.add_argument("--output", type=str, default=None, help="Output model path")

    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
