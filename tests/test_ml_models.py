import pytest
import os
from src.detectors.rf_detector import predict as rf_predict
from src.detectors.xgb_detector import predict as xgb_predict

@pytest.mark.parametrize("predict_fn", [rf_predict, xgb_predict])
def test_ml_models_exact_match(predict_fn):
    score = predict_fn("google.com", "google.com")
    assert score < 0.5, "Exact match should be benign"

@pytest.mark.parametrize("predict_fn", [rf_predict, xgb_predict])
def test_ml_models_phishing(predict_fn):
    score = predict_fn("paypal.com", "paypal-secure-login-update.com")
    assert score >= 0.5, "Stacked keywords should be phishing"

@pytest.mark.parametrize("predict_fn", [rf_predict, xgb_predict])
def test_ml_models_homoglyph(predict_fn):
    score = predict_fn("facebook.com", "faceb00k.com")
    assert score >= 0.5, "Obvious homoglyph should be phishing"
