import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

from src.config import get_settings

settings = get_settings()

_CROSSENCODER_MODEL = None
_CROSSENCODER_TOKENIZER = None


def _load_model():
    global _CROSSENCODER_MODEL, _CROSSENCODER_TOKENIZER
    if _CROSSENCODER_MODEL is not None:
        return _CROSSENCODER_MODEL, _CROSSENCODER_TOKENIZER

    model_path = settings.CROSSENCODER_MODEL_PATH

    if os.path.exists(model_path) and os.listdir(model_path):
        _CROSSENCODER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _CROSSENCODER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
    else:
        _CROSSENCODER_TOKENIZER = AutoTokenizer.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
        _CROSSENCODER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L6-v2", num_labels=2
        )

    if torch.cuda.is_available():
        _CROSSENCODER_MODEL = _CROSSENCODER_MODEL.cuda()
    elif torch.backends.mps.is_available():
        _CROSSENCODER_MODEL = _CROSSENCODER_MODEL.to("mps")

    _CROSSENCODER_MODEL.eval()

    return _CROSSENCODER_MODEL, _CROSSENCODER_TOKENIZER


def predict(orig: str, susp: str) -> float:
    """
    CrossEncoder phishing detection.

    Uses a cross-encoder to directly classify URL pairs as phishing or benign.

    Args:
        orig: Original/legitimate URL/domain
        susp: Suspicious URL/domain to check

    Returns:
        float: Probability of being phishing (0-1)
    """
    model, tokenizer = _load_model()

    orig_clean = orig.split(".")[0].lower()
    susp_clean = susp.split(".")[0].lower()

    sentence_pairs = [[orig_clean, susp_clean]]

    features = tokenizer(
        sentence_pairs,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        features = {k: v.cuda() for k, v in features.items()}
    elif torch.backends.mps.is_available():
        features = {k: v.to("mps") for k, v in features.items()}

    with torch.no_grad():
        outputs = model(**features)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        phishing_prob = probs[0][1].item()

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


def predict_batch(orig_list: list, susp_list: list) -> list:
    """
    Predict multiple URL pairs at once for efficiency.

    Args:
        orig_list: List of original URLs
        susp_list: List of suspicious URLs

    Returns:
        list: List of phishing probabilities
    """
    model, tokenizer = _load_model()

    orig_clean_list = [orig.split(".")[0].lower() for orig in orig_list]
    susp_clean_list = [susp.split(".")[0].lower() for susp in susp_list]

    sentence_pairs = [
        [orig, susp] for orig, susp in zip(orig_clean_list, susp_clean_list)
    ]

    features = tokenizer(
        sentence_pairs,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        features = {k: v.cuda() for k, v in features.items()}
    elif torch.backends.mps.is_available():
        features = {k: v.to("mps") for k, v in features.items()}

    with torch.no_grad():
        outputs = model(**features)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    phishing_probs = probs[:, 1].cpu().tolist()

    return phishing_probs
