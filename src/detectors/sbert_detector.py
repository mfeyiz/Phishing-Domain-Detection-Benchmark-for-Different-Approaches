import os
import torch
import json
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
)
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.config import get_settings

settings = get_settings()

_SBERT_MODEL = None
_SBERT_TRAINED = False


def _load_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL

    model_path = settings.SBERT_MODEL_PATH

    if os.path.exists(model_path) and os.path.isdir(model_path):
        _SBERT_MODEL = SentenceTransformer(model_path)
    else:
        _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if torch.cuda.is_available():
        _SBERT_MODEL = _SBERT_MODEL.to("cuda")
    elif torch.backends.mps.is_available():
        _SBERT_MODEL = _SBERT_MODEL.to("mps")

    return _SBERT_MODEL


def predict(orig: str, susp: str) -> float:
    """
    SBERT Bi-Encoder phishing detection.

    Uses cosine similarity between URL embeddings:
    - High similarity + different domains = phishing
    - Low similarity = benign

    Args:
        orig: Original/legitimate URL/domain
        susp: Suspicious URL/domain to check

    Returns:
        float: Probability of being phishing (0-1)
    """
    model = _load_model()

    orig_clean = orig.split(".")[0].lower()
    susp_clean = susp.split(".")[0].lower()

    embeddings = model.encode([orig_clean, susp_clean], convert_to_tensor=True)

    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
    elif torch.backends.mps.is_available():
        embeddings = embeddings.to("mps")

    orig_emb = embeddings[0].unsqueeze(0)
    susp_emb = embeddings[1].unsqueeze(0)

    similarity = cosine_similarity(orig_emb.cpu().numpy(), susp_emb.cpu().numpy())[0][0]

    domain_same = orig_clean == susp_clean

    if domain_same:
        phishing_prob = 1.0 - similarity
    else:
        phishing_prob = similarity

    phishing_prob = max(0.0, min(1.0, phishing_prob))

    return float(phishing_prob)


def predict_with_label(orig: str, susp: str, threshold: float = 0.5) -> tuple:
    """
    Predict with label.

    Returns:
        tuple: (label, probability)
    """
    prob = predict(orig, susp)

    if prob >= threshold:
        label = "Phishing"
    else:
        label = "Temiz"

    return label, prob


def retrain(
    train_data_path: str = "data/train.json",
    val_data_path: str = "data/val.json",
    epochs: int = 3,
    batch_size: int = 32,
    save: bool = True,
):
    """
    Fine-tune SBERT model using contrastive learning.

    Phishing pairs should have high similarity (pull together),
    benign pairs should have low similarity (push apart).
    """
    global _SBERT_MODEL, _SBERT_TRAINED

    print("Loading training data...")
    with open(train_data_path, "r") as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} training samples")

    train_examples = []
    for item in train_data:
        orig = item["original"].split(".")[0].lower()
        susp = item["suspicious"].split(".")[0].lower()
        label = item["label"]

        example = InputExample(texts=[orig, susp], label=1.0 if label == 1 else 0.0)
        train_examples.append(example)

    print(f"Created {len(train_examples)} training examples")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    base_model = models.Transformer("sentence-transformers/all-MiniLM-L6-v2")
    pooling_model = models.Pooling(
        base_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    _SBERT_MODEL = SentenceTransformer(modules=[base_model, pooling_model])

    if torch.cuda.is_available():
        _SBERT_MODEL = _SBERT_MODEL.to("cuda")
    elif torch.backends.mps.is_available():
        _SBERT_MODEL = _SBERT_MODEL.to("mps")

    train_loss = losses.ContrastiveLoss(model=_SBERT_MODEL)

    print(f"Training SBERT for {epochs} epochs...")
    _SBERT_MODEL.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True,
    )

    _SBERT_TRAINED = True

    if save:
        model_path = settings.SBERT_MODEL_PATH
        os.makedirs(model_path, exist_ok=True)
        _SBERT_MODEL.save(model_path)
        print(f"Model saved to {model_path}")

    return _SBERT_MODEL


def evaluate(test_data_path: str = "data/test.json", threshold: float = 0.5):
    """
    Evaluate model on test data.
    """
    model = _load_model()

    print("Loading test data...")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    print(f"Evaluating on {len(test_data)} samples...")

    tp = tn = fp = fn = 0

    for item in test_data:
        orig = item["original"]
        susp = item["suspicious"]
        true_label = item["label"]

        pred_prob = predict(orig, susp)
        pred_label = 1 if pred_prob >= threshold else 0

        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(test_data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  TP/TN/FP/FN: {tp}/{tn}/{fp}/{fn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
