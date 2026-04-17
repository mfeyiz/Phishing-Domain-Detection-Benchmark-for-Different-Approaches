#!/usr/bin/env python3
"""
Benchmark tool for evaluating phishing detection models.
This script compares different detection approaches against a dataset.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import time

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.generators.hard import HardGenerator
from src.detectors.algorithmic import predict as algorithmic_predict
from src.detectors.rf_detector import predict as rf_predict
from src.detectors.xgb_detector import predict as xgb_predict
from src.detectors.sbert_detector import predict as sbert_predict
from src.detectors.urlbert_detector import predict as urlbert_predict
from src.detectors.crossencoder_detector import predict as crossencoder_predict

def evaluate_model(name, predict_fn, dataset, threshold=0.5, is_algorithmic=False):
    """Evaluate a specific model on a dataset."""
    print(f"    Evaluating {name}...")
    tp = tn = fp = fn = 0
    start_time = time.time()
    
    for item in dataset:
        orig = item["orig"]
        susp = item["susp"]
        true_label = item["label"]
        
        try:
            if is_algorithmic:
                result = predict_fn(orig, susp)
                score = result[1] if isinstance(result, tuple) else result
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
            print(f"      Error evaluating {orig} vs {susp}: {e}")
            
    latency = (time.time() - start_time) / len(dataset) * 1000
    accuracy = (tp + tn) / len(dataset) if dataset else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "method": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def main():
    parser = argparse.ArgumentParser(description="Phishing Detection Benchmark")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="JSON output file")
    parser.add_argument("--output-md", type=str, default="benchmark_results.md", help="Markdown output file")
    
    args = parser.parse_args()
    
    print("\n[1] Generating benchmark dataset (Hard Examples)...")
    gen = HardGenerator(seed=42)
    test_data = gen.generate_dataset(args.samples)
    
    phishing_count = sum(1 for d in test_data if d["label"] == 1)
    benign_count = len(test_data) - phishing_count
    print(f"    Total: {len(test_data)} | Phishing: {phishing_count} | Benign: {benign_count}")
    
    print("\n[2] Running benchmarks...")
    models = [
        ("Algorithmic", algorithmic_predict, True),
        ("Random Forest", rf_predict, False),
        ("XGBoost", xgb_predict, False),
        ("SBERT", sbert_predict, False),
        ("URLBERT", urlbert_predict, False),
        ("CrossEncoder", crossencoder_predict, False)
    ]
    
    results = []
    for name, fn, is_alg in models:
        metrics = evaluate_model(name, fn, test_data, args.threshold, is_alg)
        results.append(metrics)
        
    # Save to JSON
    summary = {
        "generated_at": datetime.now().isoformat(),
        "threshold": args.threshold,
        "dataset": {"total": args.samples, "phishing": phishing_count, "benign": benign_count},
        "results": results
    }
    
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Generate Markdown
    with open(args.output_md, "w") as f:
        f.write("# Phishing Detection Benchmark Results\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Samples: {args.samples}\n\n")
        f.write("| Method | Accuracy | Precision | Recall | F1 Score | Latency (ms) |\n")
        f.write("|--------|----------|-----------|--------|----------|--------------|\n")
        for r in results:
            f.write(f"| {r['method']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['latency_ms']:.2f} |\n")
            
    print(f"\n[3] Done! Results saved to {args.output_json} and {args.output_md}")

if __name__ == "__main__":
    main()
