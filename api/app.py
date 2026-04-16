import os
import time
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import HTTPException

from src.config import get_settings
from src.utils import extract_features
from api import db

app = Flask(__name__, template_folder="templates")

settings = get_settings()

_algorithmic_model = None
_rf_model = None
_xgb_model = None
_urlbert_model = None
_sbert_model = None
_crossencoder_model = None

start_time = time.time()

models_loaded = {}


def _lazy_load_algorithmic():
    global _algorithmic_model, models_loaded
    if _algorithmic_model is None:
        from src.detectors.algorithmic import predict as _predict

        start = time.time()
        _algorithmic_model = _predict
        models_loaded["algorithmic"] = time.time() - start
    return _algorithmic_model


def _lazy_load_rf():
    global _rf_model, models_loaded
    if _rf_model is None:
        from src.detectors.rf_detector import predict as _predict

        start = time.time()
        _rf_model = _predict
        models_loaded["rf"] = time.time() - start
    return _rf_model


def _lazy_load_xgb():
    global _xgb_model, models_loaded
    if _xgb_model is None:
        from src.detectors.xgb_detector import predict as _predict

        start = time.time()
        _xgb_model = _predict
        models_loaded["xgb"] = time.time() - start
    return _xgb_model


def _lazy_load_urlbert():
    global _urlbert_model, models_loaded
    if _urlbert_model is None:
        from src.detectors.urlbert_detector import predict as _predict

        start = time.time()
        _urlbert_model = _predict
        models_loaded["urlbert"] = time.time() - start
    return _urlbert_model


def _lazy_load_sbert():
    global _sbert_model, models_loaded
    if _sbert_model is None:
        from src.detectors.sbert_detector import predict as _predict

        start = time.time()
        _sbert_model = _predict
        models_loaded["sbert"] = time.time() - start
    return _sbert_model


def _lazy_load_crossencoder():
    global _crossencoder_model, models_loaded
    if _crossencoder_model is None:
        from src.detectors.crossencoder_detector import predict as _predict

        start = time.time()
        _crossencoder_model = _predict
        models_loaded["crossencoder"] = time.time() - start
    return _crossencoder_model


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    db_status = "connected"
    try:
        with db.get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        db_status = "disconnected"

    return jsonify(
        {
            "status": "healthy",
            "models_loaded": models_loaded,
            "database": db_status,
            "uptime_seconds": time.time() - start_time,
        }
    )


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify(
        [
            {"name": name, "available": name in models_loaded}
            for name in ["algorithmic", "rf", "xgb", "urlbert", "sbert", "crossencoder"]
        ]
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    original = data.get("original")
    suspicious = data.get("suspicious")
    method = data.get("method", "algorithmic")
    threshold = float(data.get("threshold", 0.5))

    if not original or not suspicious:
        return jsonify({"error": "Missing 'original' or 'suspicious' field"}), 400

    try:
        if method == "algorithmic":
            predict_fn = _lazy_load_algorithmic()
            result = predict_fn(original, suspicious)

            if isinstance(result, tuple):
                label, confidence = result
            else:
                label = "Phishing" if result > threshold else "Temiz"
                confidence = result if isinstance(result, float) else 0.5

        elif method == "rf":
            predict_fn = _lazy_load_rf()
            confidence = predict_fn(original, suspicious)
            label = "Phishing" if confidence >= threshold else "Temiz"

        elif method == "xgb":
            predict_fn = _lazy_load_xgb()
            confidence = predict_fn(original, suspicious)
            label = "Phishing" if confidence >= threshold else "Temiz"

        elif method == "urlbert":
            predict_fn = _lazy_load_urlbert()
            confidence = predict_fn(original, suspicious)
            label = "Phishing" if confidence >= threshold else "Temiz"

        elif method == "sbert":
            predict_fn = _lazy_load_sbert()
            confidence = predict_fn(original, suspicious)
            label = "Phishing" if confidence >= threshold else "Temiz"

        elif method == "crossencoder":
            predict_fn = _lazy_load_crossencoder()
            confidence = predict_fn(original, suspicious)
            label = "Phishing" if confidence >= threshold else "Temiz"

        else:
            return jsonify({"error": f"Unknown method: {method}"}), 400

        is_phishing = "phishing" in label.lower() or confidence >= threshold

        prediction_id = None
        try:
            prediction_id = db.save_prediction(
                original, suspicious, method, is_phishing, confidence, label
            )
        except Exception as e:
            print(f"Failed to save prediction to DB: {e}")

        return jsonify(
            {
                "original": original,
                "suspicious": suspicious,
                "method": method,
                "is_phishing": is_phishing,
                "confidence": float(confidence),
                "label": label,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    original = data.get("original")
    suspicious = data.get("suspicious")
    threshold = float(data.get("threshold", 0.5))

    if not original or not suspicious:
        return jsonify({"error": "Missing 'original' or 'suspicious' field"}), 400

    results = {}

    for method in ["algorithmic", "rf", "xgb", "urlbert", "sbert", "crossencoder"]:
        try:
            req_data = {
                "original": original,
                "suspicious": suspicious,
                "method": method,
                "threshold": threshold,
            }

            with app.test_request_context(
                json=req_data, method="POST", content_type="application/json"
            ):
                result = predict()
                results[method] = result.get_json()

        except Exception as e:
            results[method] = {"error": str(e)}

    return jsonify(results)


@app.route("/history", methods=["GET"])
def get_history():
    limit = request.args.get("limit", 50, type=int)
    try:
        history = db.get_prediction_history(limit)
        return jsonify({"history": [dict(row) for row in history]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<int:prediction_id>", methods=["GET"])
def get_history_item(prediction_id):
    try:
        item = db.get_prediction_by_id(prediction_id)
        if not item:
            return jsonify({"error": "Not found"}), 404
        return jsonify(dict(item))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        stats = db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/training/add", methods=["POST"])
def add_training_data():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    original_url = data.get("original_url")
    suspicious_url = data.get("suspicious_url")
    label = data.get("label")
    features = data.get("features")
    source = data.get("source")

    if not original_url or not suspicious_url or not label:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        training_id = db.save_training_data(
            original_url, suspicious_url, label, features, source
        )
        return jsonify({"id": training_id, "status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        db.init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")

    app.run(host="0.0.0.0", port=8000, debug=False)
