import pytest
from src.detectors.algorithmic import predict

def test_algorithmic_exact_match():
    label, score = predict("google.com", "google.com")
    assert label == "Temiz"
    assert score < 0.1

def test_algorithmic_homoglyph():
    label, score = predict("google.com", "googIe.com")
    assert label == "Phishing"

def test_algorithmic_typosquatting():
    label, score = predict("facebook.com", "faceb00k.com")
    assert label == "Phishing"

def test_algorithmic_brand_variant():
    label, score = predict("paypal.com", "paypal-login-secure.com")
    assert label == "Phishing"
