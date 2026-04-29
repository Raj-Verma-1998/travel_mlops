"""
tests/test_api.py
==================
Unit tests for the Travel ML REST API and model inference.
Run with:  pytest tests/ -v
"""

import os, sys, json
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

@pytest.fixture(scope="module")
def sample_flight():
    return {
        "from":       "Recife (PE)",
        "to":         "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency":     "FlyingDrops",
        "time":       1.76,
        "distance":   676.53,
        "year":       2019,
        "month":      9,
        "dayofweek":  3,
    }

@pytest.fixture(scope="module")
def sample_user_features():
    return {
        "age": 35,
        "company_enc": 1,
        "flight_count": 12,
        "avg_flight_price": 950.0,
        "total_flight_spend": 11400.0,
        "avg_flight_time": 2.5,
        "avg_distance": 800.0,
        "max_flight_price": 1500.0,
        "min_flight_price": 400.0,
        "flights_economic": 4,
        "flights_firstClass": 6,
        "flights_premium": 2,
        "agency_CloudFy": 3,
        "agency_FlyingDrops": 5,
        "agency_Rainbow": 4,
        "hotel_count": 8,
        "avg_hotel_days": 3.0,
        "avg_hotel_price_day": 280.0,
        "avg_hotel_total": 840.0,
        "total_hotel_spend": 6720.0,
    }


# ─────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 206)

    def test_health_json_structure(self, client):
        data = client.get("/health").get_json()
        assert "status" in data
        assert "models_loaded" in data


# ─────────────────────────────────────────────────────────────
# Flight price prediction
# ─────────────────────────────────────────────────────────────
class TestFlightPrice:
    def test_prediction_returns_200(self, client, sample_flight):
        resp = client.post("/predict/flight-price",
                           json=sample_flight)
        assert resp.status_code == 200

    def test_prediction_has_price_key(self, client, sample_flight):
        data = client.post("/predict/flight-price",
                           json=sample_flight).get_json()
        assert "predicted_price" in data

    def test_prediction_price_is_positive(self, client, sample_flight):
        data = client.post("/predict/flight-price",
                           json=sample_flight).get_json()
        assert data["predicted_price"] > 0

    def test_prediction_price_in_realistic_range(self, client, sample_flight):
        data = client.post("/predict/flight-price",
                           json=sample_flight).get_json()
        # Dataset price range: $301 – $1,754
        assert 200 < data["predicted_price"] < 2500

    def test_empty_payload_returns_400_or_200(self, client):
        resp = client.post("/predict/flight-price", json={})
        assert resp.status_code in (200, 400)


# ─────────────────────────────────────────────────────────────
# Gender classification
# ─────────────────────────────────────────────────────────────
class TestGenderClassification:
    def test_prediction_returns_200(self, client, sample_user_features):
        resp = client.post("/predict/gender", json=sample_user_features)
        assert resp.status_code == 200

    def test_prediction_has_gender_key(self, client, sample_user_features):
        data = client.post("/predict/gender",
                           json=sample_user_features).get_json()
        assert "predicted_gender" in data

    def test_predicted_gender_is_valid(self, client, sample_user_features):
        data = client.post("/predict/gender",
                           json=sample_user_features).get_json()
        assert data["predicted_gender"] in ("male", "female", "none")

    def test_confidence_between_0_and_1(self, client, sample_user_features):
        data = client.post("/predict/gender",
                           json=sample_user_features).get_json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_probabilities_sum_to_1(self, client, sample_user_features):
        data = client.post("/predict/gender",
                           json=sample_user_features).get_json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────
# Hotel recommendations
# ─────────────────────────────────────────────────────────────
class TestHotelRecommendation:
    def test_returns_200(self, client):
        resp = client.post("/recommend/hotels",
                           json={"user_code": 42, "top_k": 3})
        assert resp.status_code == 200

    def test_has_recommendations_key(self, client):
        data = client.post("/recommend/hotels",
                           json={"user_code": 42, "top_k": 3}).get_json()
        assert "recommendations" in data

    def test_returns_correct_count(self, client):
        data = client.post("/recommend/hotels",
                           json={"user_code": 42, "top_k": 3}).get_json()
        assert len(data["recommendations"]) <= 3

    def test_cold_start_user(self, client):
        """User code 9999 does not exist → cold-start fallback."""
        resp = client.post("/recommend/hotels",
                           json={"user_code": 9999, "top_k": 5})
        assert resp.status_code == 200

    def test_recommendation_has_hotel_name(self, client):
        data = client.post("/recommend/hotels",
                           json={"user_code": 42, "top_k": 2}).get_json()
        for rec in data["recommendations"]:
            assert "hotel" in rec


# ─────────────────────────────────────────────────────────────
# Model artifacts tests
# ─────────────────────────────────────────────────────────────
class TestModelArtifacts:
    MODEL_DIR = os.path.join(ROOT, "models", "artifacts")

    def test_regression_model_exists(self):
        path = os.path.join(self.MODEL_DIR, "flight_price_model.pkl")
        assert os.path.exists(path), "Regression model artifact missing"

    def test_classification_model_exists(self):
        path = os.path.join(self.MODEL_DIR, "gender_model.pkl")
        assert os.path.exists(path), "Classification model artifact missing"

    def test_recommendation_artifacts_exist(self):
        path = os.path.join(self.MODEL_DIR, "recommendation_artifacts.pkl")
        assert os.path.exists(path), "Recommendation artifacts missing"

    def test_regression_metrics_threshold(self):
        path = os.path.join(self.MODEL_DIR, "flight_metrics.json")
        assert os.path.exists(path), "Metrics JSON missing"
        with open(path) as f:
            m = json.load(f)
        assert m["R2"]   >= 0.85, f"R2 {m['R2']} below 0.85"
        assert m["RMSE"] <= 100,  f"RMSE {m['RMSE']} above 100"

    def test_classification_metrics_threshold(self):
        path = os.path.join(self.MODEL_DIR, "gender_metrics.json")
        if not os.path.exists(path):
            pytest.skip("Gender metrics file not found")
        with open(path) as f:
            m = json.load(f)
        assert m["accuracy"] >= 0.30, f"Accuracy {m['accuracy']} below 0.30"


# ─────────────────────────────────────────────────────────────
# Data quality tests
# ─────────────────────────────────────────────────────────────
class TestDataQuality:
    DATA_DIR = os.path.join(ROOT, "data")

    def test_flights_no_null_price(self):
        df = pd.read_csv(os.path.join(self.DATA_DIR, "flights.csv"))
        assert df["price"].isnull().sum() == 0

    def test_flights_price_positive(self):
        df = pd.read_csv(os.path.join(self.DATA_DIR, "flights.csv"))
        assert (df["price"] > 0).all()

    def test_users_gender_values(self):
        df = pd.read_csv(os.path.join(self.DATA_DIR, "users.csv"))
        valid = {"male", "female", "none"}
        assert set(df["gender"].unique()).issubset(valid)

    def test_hotels_total_equals_days_times_price(self):
        df = pd.read_csv(os.path.join(self.DATA_DIR, "hotels.csv"))
        computed = (df["days"] * df["price"]).round(2)
        diff     = (computed - df["total"]).abs()
        assert (diff < 1.0).mean() > 0.99   # 99%+ rows consistent
