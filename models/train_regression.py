"""
Flight Price Regression Model
==============================
Predicts flight ticket price using XGBoost.
Tracks experiments with MLflow.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "flights.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "artifacts")


# ─────────────────────────────────────────────────────────────
# 1.  Data Loading
# ─────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows | {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2.  Feature Engineering & Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop identifier columns – not predictive
    df.drop(columns=["travelCode", "userCode"], errors="ignore", inplace=True)

    # Date → temporal features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
        df["year"]      = df["date"].dt.year
        df["month"]     = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df.drop(columns=["date"], inplace=True)

    # Price per km (proxy for cabin premium)
    if "price" in df.columns and "distance" in df.columns:
        df["price_per_km"] = df["price"] / (df["distance"] + 1e-6)

    # Speed proxy
    if "distance" in df.columns and "time" in df.columns:
        df["speed"] = df["distance"] / (df["time"] + 1e-6)

    # Encode categoricals
    cat_cols = ["from", "to", "flightType", "agency"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Separate target
    X = df.drop(columns=["price"])
    y = df["price"]

    feature_names = list(X.columns)

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    return X_scaled, y, encoders, scaler, feature_names


# ─────────────────────────────────────────────────────────────
# 3.  Model Training
# ─────────────────────────────────────────────────────────────
def train(data_path: str = DATA_PATH, model_dir: str = MODEL_DIR) -> dict:
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)
    X, y, encoders, scaler, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.80,
        "colsample_bytree": 0.80,
        "random_state":     42,
        "n_jobs":           -1,
    }

    # ── MLflow tracking ──────────────────────────────────────
    mlflow.set_experiment("flight_price_regression")
    with mlflow.start_run(run_name="xgboost_v1"):
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)

        metrics = {"MAE": round(mae, 4),
                   "RMSE": round(rmse, 4),
                   "R2": round(r2, 4),
                   "MAPE": round(mape, 4)}

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "flight_price_model")

        print("\n[RESULTS] Regression Metrics")
        print(f"  MAE  : {mae:.2f}")
        print(f"  RMSE : {rmse:.2f}")
        print(f"  R²   : {r2:.4f}")
        print(f"  MAPE : {mape:.2f}%")

        # ── Save artifacts ──
        joblib.dump(model,    os.path.join(model_dir, "flight_price_model.pkl"))
        joblib.dump(scaler,   os.path.join(model_dir, "flight_scaler.pkl"))
        joblib.dump(encoders, os.path.join(model_dir, "flight_encoders.pkl"))
        with open(os.path.join(model_dir, "flight_feature_names.json"), "w") as f:
            json.dump(feature_names, f)
        with open(os.path.join(model_dir, "flight_metrics.json"), "w") as f:
            json.dump(metrics, f)

        # Feature importance
        fi = pd.Series(model.feature_importances_, index=feature_names)
        fi_sorted = fi.sort_values(ascending=False)
        print("\n[FEATURE IMPORTANCE]")
        print(fi_sorted.to_string())

    print(f"\n[INFO] Artifacts saved → {model_dir}")
    return metrics


if __name__ == "__main__":
    train()
