import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils import extract_features
from src.config import get_settings

settings = get_settings()

_RF_MODEL = None


def _load_model():
    global _RF_MODEL
    if _RF_MODEL is not None:
        return _RF_MODEL

    model_path = settings.RF_MODEL_PATH

    if os.path.exists(model_path):
        import joblib

        _RF_MODEL = joblib.load(model_path)
        return _RF_MODEL

    raise FileNotFoundError(f"Model not found: {model_path}")


def predict(orig: str, susp: str) -> float:
    """
    Random Forest phishing detection.

    Returns:
        float: Probability of being phishing (0-1)
    """
    model = _load_model()
    feats = extract_features(orig, susp)
    prob = model.predict_proba([list(feats.values())])[0][1]
    return float(prob)


def retrain(n_samples: int = 1200, save: bool = True):
    """Retrain the model with new data."""
    from src.generators import SimpleGenerator

    gen = SimpleGenerator()
    data = gen.generate_dataset(n_samples)

    X = []
    y = []
    for item in data:
        feats = extract_features(item["orig"], item["susp"])
        X.append(list(feats.values()))
        y.append(item["label"])

    model = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
    model.fit(X, y)

    if save:
        import joblib

        joblib.dump(model, settings.RF_MODEL_PATH)

    global _RF_MODEL
    _RF_MODEL = model

    return model
