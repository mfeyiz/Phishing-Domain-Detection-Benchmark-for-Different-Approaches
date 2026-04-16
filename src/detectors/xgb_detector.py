import os
import numpy as np

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.utils import extract_features
from src.config import get_settings

settings = get_settings()

_XGB_MODEL = None


def _load_model():
    global _XGB_MODEL
    if _XGB_MODEL is not None:
        return _XGB_MODEL

    model_path = settings.XGB_MODEL_PATH

    if os.path.exists(model_path):
        import joblib

        _XGB_MODEL = joblib.load(model_path)
        return _XGB_MODEL

    raise FileNotFoundError(f"Model not found: {model_path}")


def predict(orig: str, susp: str) -> float:
    """
    XGBoost phishing detection.

    Returns:
        float: Probability of being phishing (0-1)
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not installed")

    model = _load_model()
    feats = extract_features(orig, susp)
    prob = model.predict_proba([list(feats.values())])[0][1]
    return float(prob)


def retrain(n_samples: int = 1200, save: bool = True):
    """Retrain the model with new data."""
    from src.generators import SimpleGenerator

    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not installed")

    gen = SimpleGenerator()
    data = gen.generate_dataset(n_samples)

    X = []
    y = []
    for item in data:
        feats = extract_features(item["orig"], item["susp"])
        X.append(list(feats.values()))
        y.append(item["label"])

    model = XGBClassifier(
        eval_metric="logloss",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X, y)

    if save:
        import joblib

        joblib.dump(model, settings.XGB_MODEL_PATH)

    global _XGB_MODEL
    _XGB_MODEL = model

    return model
