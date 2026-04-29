"""
Gender Classification Model
============================
Classifies user gender (male / female / none) from aggregated
flight + hotel behavioural features.
Tracked with MLflow.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ─────────────────────────────────────────────────────────────
# 1.  Load & merge datasets
# ─────────────────────────────────────────────────────────────
def load_and_merge() -> pd.DataFrame:
    users   = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))

    # ── Flight aggregates per user ──
    f_agg = flights.groupby("userCode").agg(
        flight_count        = ("travelCode", "count"),
        avg_flight_price    = ("price",      "mean"),
        total_flight_spend  = ("price",      "sum"),
        avg_flight_time     = ("time",       "mean"),
        avg_distance        = ("distance",   "mean"),
        max_flight_price    = ("price",      "max"),
        min_flight_price    = ("price",      "min"),
    ).reset_index()

    # Flight-type distribution
    ftype = (flights.groupby(["userCode", "flightType"])
             .size().unstack(fill_value=0)
             .rename(columns=lambda c: f"flights_{c}"))
    ftype.reset_index(inplace=True)

    # Agency preference
    agency = (flights.groupby(["userCode", "agency"])
              .size().unstack(fill_value=0)
              .rename(columns=lambda c: f"agency_{c}"))
    agency.reset_index(inplace=True)

    # ── Hotel aggregates per user ──
    h_agg = hotels.groupby("userCode").agg(
        hotel_count         = ("travelCode", "count"),
        avg_hotel_days      = ("days",       "mean"),
        avg_hotel_price_day = ("price",      "mean"),
        avg_hotel_total     = ("total",      "mean"),
        total_hotel_spend   = ("total",      "sum"),
    ).reset_index()

    # ── Merge all ──
    # Drop userCode from aggregates before merging to avoid column conflicts
    f_agg.rename(columns={"userCode": "userCode_f"}, inplace=True)
    h_agg.rename(columns={"userCode": "userCode_h"}, inplace=True)
    if "userCode" in ftype.columns:  ftype.rename(columns={"userCode": "userCode_ft"}, inplace=True)
    if "userCode" in agency.columns: agency.rename(columns={"userCode": "userCode_ag"}, inplace=True)

    df = users.merge(f_agg,  left_on="code", right_on="userCode_f",  how="left")
    df = df.merge(ftype,     left_on="code", right_on="userCode_ft", how="left")
    df = df.merge(agency,    left_on="code", right_on="userCode_ag", how="left")
    df = df.merge(h_agg,     left_on="code", right_on="userCode_h",  how="left")
    df.fillna(0, inplace=True)

    print(f"[INFO] Merged dataset: {df.shape[0]} users × {df.shape[1]} features")
    return df


# ─────────────────────────────────────────────────────────────
# 2.  Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Encode company
    le_company = LabelEncoder()
    if "company" in df.columns:
        df["company_enc"] = le_company.fit_transform(df["company"].astype(str))

    # Drop non-feature columns
    drop_cols = ["code", "name", "company", "gender"] + \
                [c for c in df.columns if c.startswith("userCode")]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Target
    le_gender = LabelEncoder()
    # already extracted before drop
    return df, le_gender, le_company


def build_Xy(df_raw: pd.DataFrame):
    df = df_raw.copy()

    le_company = LabelEncoder()
    if "company" in df.columns:
        df["company_enc"] = le_company.fit_transform(df["company"].astype(str))

    le_gender = LabelEncoder()
    y = le_gender.fit_transform(df["gender"].astype(str))

    drop_cols = ["code", "name", "company", "gender"] + \
                [c for c in df.columns if c.startswith("userCode")]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    feature_names = list(df.columns)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df), columns=feature_names)

    return X, y, le_gender, scaler, feature_names


# ─────────────────────────────────────────────────────────────
# 3.  Train
# ─────────────────────────────────────────────────────────────
def train(model_dir: str = MODEL_DIR) -> dict:
    os.makedirs(model_dir, exist_ok=True)

    df = load_and_merge()
    X, y, le_gender, scaler, feature_names = build_Xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    params = {
        "n_estimators":  200,
        "learning_rate": 0.05,
        "max_depth":     4,
        "subsample":     0.8,
        "random_state":  42,
    }

    mlflow.set_experiment("gender_classification")
    with mlflow.start_run(run_name="gradient_boosting_v1"):
        mlflow.log_params(params)

        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred     = clf.predict(X_test)
        y_prob     = clf.predict_proba(X_test)
        acc        = accuracy_score(y_test, y_pred)
        # OvR AUC for multiclass
        auc        = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        cv_scores  = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        metrics = {
            "accuracy":    round(acc,  4),
            "roc_auc":     round(auc,  4),
            "cv_mean":     round(cv_scores.mean(), 4),
            "cv_std":      round(cv_scores.std(),  4),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "gender_model")

        print("\n[RESULTS] Classification Metrics")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(y_test, y_pred,
                                    target_names=le_gender.classes_))

        # ── Save artifacts ──
        joblib.dump(clf,          os.path.join(model_dir, "gender_model.pkl"))
        joblib.dump(scaler,       os.path.join(model_dir, "gender_scaler.pkl"))
        joblib.dump(le_gender,    os.path.join(model_dir, "gender_encoder.pkl"))
        with open(os.path.join(model_dir, "gender_feature_names.json"), "w") as f:
            json.dump(feature_names, f)
        with open(os.path.join(model_dir, "gender_metrics.json"), "w") as f:
            json.dump(metrics, f)

    print(f"\n[INFO] Artifacts saved → {model_dir}")
    return metrics


if __name__ == "__main__":
    train()
