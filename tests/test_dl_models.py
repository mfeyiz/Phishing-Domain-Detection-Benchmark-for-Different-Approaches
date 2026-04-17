import pytest
from src.detectors.sbert_detector import predict as sbert_predict
from src.detectors.urlbert_detector import predict as urlbert_predict
from src.detectors.crossencoder_detector import predict as crossencoder_predict

DL_MODELS = [
    sbert_predict,
    pytest.param(urlbert_predict, marks=pytest.mark.xfail(reason="URLBERT missing classification weights on HF Hub")),
    crossencoder_predict
]

@pytest.mark.parametrize("predict_fn", DL_MODELS)
def test_dl_exact_match(predict_fn):
    # Tests that the DL model does not flag identical domains as phishing
    score = predict_fn("google.com", "google.com")
    assert score < 0.5, f"Exact match should be benign, got {score}"

@pytest.mark.parametrize("predict_fn", DL_MODELS)
def test_dl_high_confidence_phishing(predict_fn):
    # Tests a severely malicious pattern that all baseline DL models should catch
    score = predict_fn("paypal.com", "paypal-secure-login-update.com")
    assert score >= 0.5, f"Expected > 0.5 for stacked keywords, got {score}"

