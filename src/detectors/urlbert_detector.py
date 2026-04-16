import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple

from src.config import get_settings

settings = get_settings()

_URLBERT_MODEL = None
_URLBERT_TOKENIZER = None


def _load_model():
    global _URLBERT_MODEL, _URLBERT_TOKENIZER
    if _URLBERT_MODEL is not None:
        return _URLBERT_MODEL, _URLBERT_TOKENIZER

    model_name = settings.URLBERT_MODEL

    _URLBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _URLBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)

    if torch.cuda.is_available():
        _URLBERT_MODEL = _URLBERT_MODEL.cuda()
    elif torch.backends.mps.is_available():
        _URLBERT_MODEL = _URLBERT_MODEL.to("mps")

    _URLBERT_MODEL.eval()

    return _URLBERT_MODEL, _URLBERT_TOKENIZER


def predict(orig: str, susp: str) -> float:
    """
    URLBERT phishing detection using pre-trained DomURLs_BERT model.

    Args:
        orig: Original/legitimate URL/domain
        susp: Suspicious URL/domain to check

    Returns:
        float: Probability of being phishing (0-1)
    """
    model, tokenizer = _load_model()

    input_text = f"{orig} [SEP] {susp}"

    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        phishing_prob = probs[0][1].item()

    return float(phishing_prob)


def predict_batch(orig: str, susp: str) -> Tuple[str, float]:
    """
    Predict with label.

    Returns:
        tuple: (label, probability)
    """
    prob = predict(orig, susp)

    if prob >= 0.5:
        label = "Phishing"
    else:
        label = "Temiz"

    return label, prob
