"""
travel_ml_pipeline.py
═════════════════════
Apache Airflow DAG – Travel ML Pipeline

Orchestrates the full MLOps workflow:
  1. data_validation     → check CSVs exist and have required columns
  2. feature_engineering → compute & save derived feature sets
  3. train_regression    → retrain XGBoost flight-price model
  4. train_classification→ retrain gender classifier
  5. train_recommendation→ rebuild recommendation artifacts
  6. evaluate_models     → load saved metrics, fail if thresholds missed
  7. register_models     → log best run to MLflow Model Registry
  8. notify              → print deployment summary

Schedule: @weekly (every Sunday at 00:00 UTC)
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty  import EmptyOperator

log = logging.getLogger(__name__)

# ── Paths (adjust to your container / host mount) ─────────────
ROOT      = os.environ.get("TRAVEL_ROOT", "/app")
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models", "artifacts")

# ── Quality thresholds ─────────────────────────────────────────
THRESHOLDS = {
    "regression":     {"R2": 0.85,  "RMSE": 100.0},
    "classification": {"accuracy": 0.30},          # dataset has near-random gender
}

default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


# ─────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────

def task_data_validation(**ctx):
    """Validate that all required CSVs are present with correct columns."""
    import pandas as pd

    required = {
        "travel_Data\flights.csv": ["travelCode", "userCode", "from", "to",
                        "flightType", "price", "time", "distance", "agency", "date"],
        "travel_Data\hotels.csv":  ["travelCode", "userCode", "name", "place",
                        "days", "price", "total", "date"],
        "travel_Data\users.csv":   ["code", "company", "name", "gender", "age"],
    }

    for filename, cols in required.items():
        path = os.path.join(DATA_DIR, filename)
        assert os.path.exists(path), f"Missing file: {path}"
        df = pd.read_csv(path, nrows=5)
        missing = set(cols) - set(df.columns)
        assert not missing, f"{filename} missing columns: {missing}"
        log.info(f"✅  {filename} validated")

    log.info("All data files validated successfully.")


def task_feature_engineering(**ctx):
    """Precompute derived features and cache for downstream tasks."""
    import pandas as pd

    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))

    # Flight-level features
    flights["date"]         = pd.to_datetime(flights["date"], dayfirst=False, errors="coerce")
    flights["month"]        = flights["date"].dt.month
    flights["dayofweek"]    = flights["date"].dt.dayofweek
    flights["speed"]        = flights["distance"] / (flights["time"] + 1e-6)
    flights["price_per_km"] = flights["price"]    / (flights["distance"] + 1e-6)

    out = os.path.join(DATA_DIR, "flights_engineered.csv")
    flights.to_csv(out, index=False)
    log.info(f"Engineered flight features saved → {out}")


def task_train_regression(**ctx):
    import sys; sys.path.insert(0, ROOT)
    from models.train_regression import train
    metrics = train(
        data_path=os.path.join(DATA_DIR, "flights.csv"),
        model_dir=MODEL_DIR
    )
    ctx["ti"].xcom_push(key="regression_metrics", value=metrics)
    log.info(f"Regression metrics: {metrics}")


def task_train_classification(**ctx):
    import sys; sys.path.insert(0, ROOT)
    from models.train_classification import train
    metrics = train(model_dir=MODEL_DIR)
    ctx["ti"].xcom_push(key="classification_metrics", value=metrics)
    log.info(f"Classification metrics: {metrics}")


def task_train_recommendation(**ctx):
    import sys; sys.path.insert(0, ROOT)
    from models.train_recommendation import train
    train(model_dir=MODEL_DIR)
    log.info("Recommendation artifacts rebuilt.")


def task_evaluate_models(**ctx):
    """Load saved metrics and assert quality thresholds."""
    ti = ctx["ti"]

    reg_metrics  = ti.xcom_pull(task_ids="train_regression",    key="regression_metrics")
    clf_metrics  = ti.xcom_pull(task_ids="train_classification", key="classification_metrics")

    # Regression checks
    if reg_metrics:
        assert reg_metrics["R2"]   >= THRESHOLDS["regression"]["R2"],   \
            f"R² {reg_metrics['R2']} below threshold {THRESHOLDS['regression']['R2']}"
        assert reg_metrics["RMSE"] <= THRESHOLDS["regression"]["RMSE"], \
            f"RMSE {reg_metrics['RMSE']} above threshold {THRESHOLDS['regression']['RMSE']}"
        log.info(f"✅  Regression thresholds passed: {reg_metrics}")

    # Classification checks
    if clf_metrics:
        assert clf_metrics["accuracy"] >= THRESHOLDS["classification"]["accuracy"], \
            f"Accuracy {clf_metrics['accuracy']} below threshold"
        log.info(f"✅  Classification thresholds passed: {clf_metrics}")


def task_register_models(**ctx):
    """Register model artifacts path in a simple registry manifest."""
    import json
    registry = {
        "flight_price_model":    os.path.join(MODEL_DIR, "flight_price_model.pkl"),
        "gender_model":          os.path.join(MODEL_DIR, "gender_model.pkl"),
        "recommendation_model":  os.path.join(MODEL_DIR, "recommendation_artifacts.pkl"),
        "registered_at":         datetime.utcnow().isoformat(),
    }
    path = os.path.join(MODEL_DIR, "model_registry.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"Model registry updated → {path}")


def task_notify(**ctx):
    ti = ctx["ti"]
    reg = ti.xcom_pull(task_ids="train_regression",    key="regression_metrics")  or {}
    clf = ti.xcom_pull(task_ids="train_classification", key="classification_metrics") or {}

    summary = f"""
╔══════════════════════════════════════════╗
║      Travel ML Pipeline – Run Complete   ║
╠══════════════════════════════════════════╣
║  Regression   R²   : {reg.get('R2',   'N/A')}
║  Regression   RMSE : {reg.get('RMSE', 'N/A')}
║  Classifier   Acc  : {clf.get('accuracy', 'N/A')}
║  Classifier   AUC  : {clf.get('roc_auc', 'N/A')}
║  Artifacts  → {MODEL_DIR}
╚══════════════════════════════════════════╝
    """
    log.info(summary)
    print(summary)


# ─────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────
with DAG(
    dag_id          = "travel_ml_pipeline",
    description     = "End-to-end Travel ML training & registration pipeline",
    default_args    = default_args,
    start_date      = datetime(2024, 1, 1),
    schedule        = "@weekly",
    catchup         = False,
    max_active_runs = 1,
    tags            = ["travel", "mlops", "training"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    validate = PythonOperator(
        task_id         = "data_validation",
        python_callable = task_data_validation,
    )

    engineer = PythonOperator(
        task_id         = "feature_engineering",
        python_callable = task_feature_engineering,
    )

    train_reg = PythonOperator(
        task_id         = "train_regression",
        python_callable = task_train_regression,
    )

    train_clf = PythonOperator(
        task_id         = "train_classification",
        python_callable = task_train_classification,
    )

    train_rec = PythonOperator(
        task_id         = "train_recommendation",
        python_callable = task_train_recommendation,
    )

    evaluate = PythonOperator(
        task_id         = "evaluate_models",
        python_callable = task_evaluate_models,
    )

    register = PythonOperator(
        task_id         = "register_models",
        python_callable = task_register_models,
    )

    notify = PythonOperator(
        task_id         = "notify",
        python_callable = task_notify,
    )

    # ── Dependency graph ──────────────────────────────────────
    #
    #  start → validate → engineer ─┬─ train_reg ─┐
    #                                ├─ train_clf ─┼─ evaluate → register → notify → end
    #                                └─ train_rec ─┘
    #
    start >> validate >> engineer >> [train_reg, train_clf, train_rec]
    [train_reg, train_clf] >> evaluate >> register >> notify >> end
    train_rec >> notify
