#!/usr/bin/env python3
"""
CrossEncoder Training Script

Fine-tunes CrossEncoder model for phishing detection using binary classification.
Training data: data/train.json
Validation: data/val.json

Usage:
    python scripts/train_crossencoder.py

Output:
    models/crossencoder_model/ - Fine-tuned CrossEncoder model
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import get_settings

settings = get_settings()


class PhishingDataset(Dataset):
    """Dataset for CrossEncoder training."""

    def __init__(self, data: list, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        orig = item["original"].split(".")[0].lower()
        susp = item["suspicious"].split(".")[0].lower()

        encoding = self.tokenizer(
            orig,
            susp,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_data(data_path: str):
    """Load training data from JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def train(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    output_path: str = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_length: int = 128,
):
    """Train CrossEncoder model."""

    if output_path is None:
        output_path = settings.CROSSENCODER_MODEL_PATH

    print("=" * 60)
    print("CrossEncoder Training")
    print("=" * 60)

    print(f"\n[1] Loading data...")
    train_data = load_data(train_path)
    print(f"    Training samples: {len(train_data)}")

    if val_path and os.path.exists(val_path):
        val_data = load_data(val_path)
        print(f"    Validation samples: {len(val_data)}")
    else:
        val_data = None
        print("    Validation: None")

    print(f"\n[2] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    if torch.cuda.is_available():
        print("    Device: CUDA")
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        print("    Device: MPS (Apple Silicon)")
        model = model.to("mps")
    else:
        print("    Device: CPU")

    print(f"\n[3] Creating datasets...")
    train_dataset = PhishingDataset(train_data, tokenizer, max_length)
    print(f"    Train dataset: {len(train_dataset)}")

    if val_data:
        val_dataset = PhishingDataset(val_data, tokenizer, max_length)
        print(f"    Val dataset: {len(val_dataset)}")
    else:
        val_dataset = None

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n[4] Starting training...")
    print(f"    Epochs: {epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Learning rate: {learning_rate}")

    trainer.train()

    print(f"\n[5] Saving model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train CrossEncoder model for phishing detection"
    )
    parser.add_argument(
        "--train", type=str, default="data/train.json", help="Training data path"
    )
    parser.add_argument(
        "--val", type=str, default="data/val.json", help="Validation data path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Base model name",
    )
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        model_name=args.model,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
