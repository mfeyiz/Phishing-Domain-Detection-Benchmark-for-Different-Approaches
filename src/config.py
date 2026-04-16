import os
from functools import lru_cache


@lru_cache()
def get_settings() -> "Settings":
    return Settings()


class Settings:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
    XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
    SBERT_MODEL_PATH = os.path.join(MODEL_DIR, "sbert_model")
    CROSSENCODER_MODEL_PATH = os.path.join(MODEL_DIR, "crossencoder_model")

    URLBERT_MODEL = os.getenv("URLBERT_MODEL", "amahdaouy/DomURLs_BERT")

    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    ENABLE_ALGORITHMIC = os.getenv("ENABLE_ALGORITHMIC", "true").lower() == "true"
    ENABLE_RF = os.getenv("ENABLE_RF", "true").lower() == "true"
    ENABLE_XGB = os.getenv("ENABLE_XGB", "true").lower() == "true"
    ENABLE_URLBERT = os.getenv("ENABLE_URLBERT", "true").lower() == "true"
    ENABLE_SBERT = os.getenv("ENABLE_SBERT", "true").lower() == "true"
    ENABLE_CROSSENCODER = os.getenv("ENABLE_CROSSENCODER", "true").lower() == "true"
