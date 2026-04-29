"""
Travel ML REST API
==================
Endpoints:
  GET  /health                     → service health + loaded models
  POST /predict/flight-price       → XGBoost regression
  POST /predict/gender             → GradientBoosting classification
  POST /recommend/hotels           → hybrid recommendation
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib, json, os, sys, logging, time
from functools import wraps

# ── allow relative imports when run directly ──────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR = os.environ.get("MODEL_DIR",
            os.path.join(ROOT_DIR, "models", "artifacts"))

sys.path.insert(0, ROOT_DIR)

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Utility: request timer decorator
# ─────────────────────────────────────────────────────────────
def timed(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        log.info(f"{f.__name__} completed in {elapsed} ms")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────
# Load all model artifacts at startup
# ─────────────────────────────────────────────────────────────
ARTIFACTS = {}

def _try_load(key, path):
    try:
        ARTIFACTS[key] = joblib.load(path)
        log.info(f"✅  Loaded: {key}")
    except FileNotFoundError:
        log.warning(f"⚠️   Not found (run training first): {path}")

def _try_json(key, path):
    try:
        with open(path) as f:
            ARTIFACTS[key] = json.load(f)
    except FileNotFoundError:
        log.warning(f"⚠️   JSON not found: {path}")

def load_all():
    _try_load("flight_model",    os.path.join(MODEL_DIR, "flight_price_model.pkl"))
    _try_load("flight_scaler",   os.path.join(MODEL_DIR, "flight_scaler.pkl"))
    _try_load("flight_encoders", os.path.join(MODEL_DIR, "flight_encoders.pkl"))
    _try_json("flight_features", os.path.join(MODEL_DIR, "flight_feature_names.json"))

    _try_load("gender_model",    os.path.join(MODEL_DIR, "gender_model.pkl"))
    _try_load("gender_scaler",   os.path.join(MODEL_DIR, "gender_scaler.pkl"))
    _try_load("gender_encoder",  os.path.join(MODEL_DIR, "gender_encoder.pkl"))
    _try_json("gender_features", os.path.join(MODEL_DIR, "gender_feature_names.json"))

    _try_load("rec_artifacts",   os.path.join(MODEL_DIR, "recommendation_artifacts.pkl"))

load_all()


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    status = {k: True for k in ARTIFACTS}
    ready  = all(k in ARTIFACTS for k in
                 ["flight_model", "gender_model", "rec_artifacts"])
    return jsonify({
        "status":        "ready" if ready else "degraded",
        "models_loaded": status,
        "model_dir":     MODEL_DIR,
    }), 200 if ready else 206


# ─────────────────────────────────────────────────────────────
# POST /predict/flight-price
# ─────────────────────────────────────────────────────────────
@app.route("/predict/flight-price", methods=["POST"])
@timed
def predict_flight_price():
    """
    Request body (JSON):
    {
        "from":       "Recife (PE)",
        "to":         "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency":     "FlyingDrops",
        "time":       1.76,
        "distance":   676.53,
        "year":       2024,
        "month":      6,
        "dayofweek":  2
    }
    """
    if "flight_model" not in ARTIFACTS:
        return jsonify({"error": "Flight price model not loaded"}), 503

    data     = request.get_json(force=True)
    model    = ARTIFACTS["flight_model"]
    scaler   = ARTIFACTS["flight_scaler"]
    encoders = ARTIFACTS["flight_encoders"]
    features = ARTIFACTS["flight_features"]

    try:
        row = {}
        cat_cols = ["from", "to", "flightType", "agency"]
        for col in cat_cols:
            if col in encoders:
                val = str(data.get(col, ""))
                le  = encoders[col]
                row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
            else:
                row[col] = 0

        for feat in features:
            if feat not in row:
                row[feat] = float(data.get(feat, 0))

        # Recompute derived features if base inputs given
        dist = row.get("distance", 1)
        t    = row.get("time", 1)
        row["speed"]        = dist / (t + 1e-9)
        row["price_per_km"] = 0.0   # unknown at inference; set zero

        X_df = pd.DataFrame([row])[features]
        X_sc = scaler.transform(X_df)
        price = float(model.predict(X_sc)[0])

        return jsonify({
            "predicted_price": round(price, 2),
            "currency":        "USD",
            "model":           "XGBoost Regressor",
        })

    except Exception as e:
        log.exception("Error in flight price prediction")
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# POST /predict/gender
# ─────────────────────────────────────────────────────────────
@app.route("/predict/gender", methods=["POST"])
@timed
def predict_gender():
    """
    Request body (JSON) – pass any subset of user feature columns.
    Missing features default to 0.
    Example:
    {
        "age": 35,
        "company_enc": 1,
        "flight_count": 12,
        "avg_flight_price": 950.0
    }
    """
    if "gender_model" not in ARTIFACTS:
        return jsonify({"error": "Gender model not loaded"}), 503

    data     = request.get_json(force=True)
    model    = ARTIFACTS["gender_model"]
    scaler   = ARTIFACTS["gender_scaler"]
    le       = ARTIFACTS["gender_encoder"]
    features = ARTIFACTS["gender_features"]

    try:
        row  = {feat: float(data.get(feat, 0)) for feat in features}
        X_df = pd.DataFrame([row])[features]
        X_sc = scaler.transform(X_df)

        pred_idx  = int(model.predict(X_sc)[0])
        pred_prob = model.predict_proba(X_sc)[0]
        label     = le.inverse_transform([pred_idx])[0]

        return jsonify({
            "predicted_gender": label,
            "confidence":       round(float(max(pred_prob)), 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(le.classes_, pred_prob)
            },
            "model": "GradientBoosting Classifier",
        })

    except Exception as e:
        log.exception("Error in gender prediction")
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# POST /recommend/hotels
# ─────────────────────────────────────────────────────────────
@app.route("/recommend/hotels", methods=["POST"])
@timed
def recommend_hotels():
    """
    Request body (JSON):
    { "user_code": 42, "top_k": 5 }
    """
    if "rec_artifacts" not in ARTIFACTS:
        return jsonify({"error": "Recommendation artifacts not loaded"}), 503

    data      = request.get_json(force=True)
    user_code = int(data.get("user_code", -1))
    top_k     = min(int(data.get("top_k", 5)), 9)

    try:
        from models.train_recommendation import recommend
        recs      = recommend(user_code, ARTIFACTS["rec_artifacts"], top_k=top_k)
        profiles  = {
            h["name"]: h
            for h in ARTIFACTS["rec_artifacts"].get("hotel_profiles", [])
        }
        result = [
            {
                "rank":           i + 1,
                "hotel":          hotel,
                "avg_price_day":  round(profiles[hotel]["avg_price_day"], 2)
                                  if hotel in profiles else None,
                "avg_total":      round(profiles[hotel]["avg_total"], 2)
                                  if hotel in profiles else None,
                "avg_days":       round(profiles[hotel]["avg_days"], 1)
                                  if hotel in profiles else None,
                "place":          profiles[hotel]["place"]
                                  if hotel in profiles else None,
            }
            for i, hotel in enumerate(recs)
        ]

        return jsonify({
            "user_code":      user_code,
            "recommendations": result,
            "model":          "Hybrid Content-Based + Collaborative Filtering",
        })

    except Exception as e:
        log.exception("Error in hotel recommendation")
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
