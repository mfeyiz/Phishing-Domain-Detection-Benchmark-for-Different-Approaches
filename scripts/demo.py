#!/usr/bin/env python3
"""
Demo script demonstrating phishing detection with hard examples.
Tests all models on sophisticated attack patterns.
"""

import sys
import json
from datetime import datetime

sys.path.insert(0, "/Users/mac/Desktop/Phising")

from src.generators import SimpleGenerator, HardGenerator
from src.detectors import algorithmic_predict, rf_predict, xgb_predict


def evaluate_model(predict_fn, dataset, threshold=0.5, is_algorithmic=False):
    """Evaluate a model on the dataset."""
    tp = tn = fp = fn = 0

    for item in dataset:
        orig = item["orig"]
        susp = item["susp"]
        true_label = item["label"]

        try:
            if is_algorithmic:
                result = predict_fn(orig, susp)
                if isinstance(result, tuple):
                    _, score = result
                else:
                    score = result
            else:
                score = predict_fn(orig, susp)

            pred = 1 if score >= threshold else 0

            if pred == 1 and true_label == 1:
                tp += 1
            elif pred == 0 and true_label == 0:
                tn += 1
            elif pred == 1 and true_label == 0:
                fp += 1
            else:
                fn += 1
        except Exception as e:
            print(f"Error: {e}")

    accuracy = (tp + tn) / len(dataset) if dataset else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

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


def run_demo():
    print("=" * 70)
    print("Phishing Detection - Hard Examples Demo")
    print("=" * 70)

    print("\n[1] Generating HARD test dataset...")
    hard_gen = HardGenerator(seed=42)
    test_data = hard_gen.generate_dataset(500)

    phishing_count = sum(1 for d in test_data if d["label"] == 1)
    benign_count = len(test_data) - phishing_count
    print(
        f"    Total: {len(test_data)} | Phishing: {phishing_count} | Benign: {benign_count}"
    )

    print("\n[2] Sample hard examples:")
    for i, item in enumerate(test_data[:5]):
        label = "PHISHING" if item["label"] == 1 else "BENIGN"
        print(f"    {i + 1}. {item['orig']} -> {item['susp']} [{label}]")

    print("\n[3] Evaluating models on HARD examples...")
    print("-" * 70)

    models = [
        ("Algorithmic", algorithmic_predict, True),
        ("Random Forest", rf_predict, False),
        ("XGBoost", xgb_predict, False),
    ]

    results = []
    for name, fn, is_alg in models:
        print(f"\n    Testing {name}...")
        metrics = evaluate_model(fn, test_data, is_algorithmic=is_alg)
        results.append((name, metrics))

        print(f"    Accuracy:  {metrics['accuracy']:.2%}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall:    {metrics['recall']:.2%}")
        print(f"    F1 Score:  {metrics['f1']:.2%}")
        print(
            f"    TP/TN/FP/FN: {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for name, m in results:
        print(
            f"{name:<20} {m['accuracy']:>9.2%} {m['precision']:>9.2%} {m['recall']:>9.2%} {m['f1']:>9.2%}"
        )

    output_file = f"demo_hard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({name: m for name, m in results}, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("\n[4] Testing specific hard cases...")
    test_cases = [
        ("google.com", "googIe.com", 1),  # Unicode homoglyph
        ("paypal.com", "paypal-login-secure-update-verify.com", 1),  # Stacked keywords
        ("facebook.com", "faceb00k.com", 1),  # Multi-step typo
        ("amazon.com", "amaz0n.net", 1),  # TLD variation + homoglyph
        ("microsoft.com", "microsoft.com", 0),  # Same domain
        ("google.com", "g0ogle.org", 1),  # Homoglyph + TLD
        ("apple.com", "appIe.com", 1),  # Mixed case homoglyph
        ("netflix.com", "netf1ix-login.com", 1),  # Combined
    ]

    print("\n" + "-" * 70)
    print(
        f"{'Original':<20} {'Suspicious':<35} {'Algorithmic':<12} {'RF':<8} {'XGB':<8}"
    )
    print("-" * 70)
    for orig, susp, true_label in test_cases:
        alg_label, alg_score = algorithmic_predict(orig, susp)
        rf_score = rf_predict(orig, susp)
        xgb_score = xgb_predict(orig, susp)

        alg_pred = "P" if "Phishing" in alg_label else "B"
        rf_pred = "P" if rf_score >= 0.5 else "B"
        xgb_pred = "P" if xgb_score >= 0.5 else "B"

        true_str = "P" if true_label == 1 else "B"
        print(
            f"{orig:<20} {susp:<35} {alg_pred}({alg_score:.2f})   {rf_pred}({rf_score:.2f})  {xgb_pred}({xgb_score:.2f})  [{true_str}]"
        )


if __name__ == "__main__":
    run_demo()
