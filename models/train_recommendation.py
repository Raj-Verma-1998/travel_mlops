"""
Hotel Recommendation Model
============================
Hybrid content-based + collaborative filtering system.
Uses cosine similarity on user behaviour vectors to recommend hotels.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ─────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────
def load_data():
    users   = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    return users, hotels, flights


# ─────────────────────────────────────────────────────────────
# 2.  Build user feature matrix
# ─────────────────────────────────────────────────────────────
def build_user_matrix(users, hotels, flights):
    f_agg = flights.groupby("userCode").agg(
        flight_count       = ("travelCode", "count"),
        avg_flight_price   = ("price",      "mean"),
        avg_distance       = ("distance",   "mean"),
        total_flight_spend = ("price",      "sum"),
    ).reset_index()

    ftype = (flights.groupby(["userCode", "flightType"])
             .size().unstack(fill_value=0)
             .rename(columns=lambda c: f"ft_{c}"))
    ftype.reset_index(inplace=True)

    h_agg = hotels.groupby("userCode").agg(
        hotel_count        = ("travelCode", "count"),
        avg_stay_days      = ("days",       "mean"),
        avg_hotel_total    = ("total",      "mean"),
        total_hotel_spend  = ("total",      "sum"),
    ).reset_index()

    df = users.merge(f_agg,  left_on="code", right_on="userCode", how="left")
    df = df.merge(ftype,     left_on="code", right_on="userCode", how="left")
    df = df.merge(h_agg,     left_on="code", right_on="userCode", how="left")
    df.fillna(0, inplace=True)

    le_gender  = LabelEncoder()
    le_company = LabelEncoder()
    df["gender_enc"]  = le_gender.fit_transform(df["gender"].astype(str))
    df["company_enc"] = le_company.fit_transform(df["company"].astype(str))

    user_codes = df["code"].values
    drop_cols  = ["code", "name", "gender", "company"] + \
                 [c for c in df.columns if c.startswith("userCode")]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df)

    return matrix, user_codes, list(df.columns)


# ─────────────────────────────────────────────────────────────
# 3.  Build hotel feature matrix
# ─────────────────────────────────────────────────────────────
def build_hotel_matrix(hotels):
    le_name  = LabelEncoder()
    le_place = LabelEncoder()

    h = hotels.copy()
    h["hotel_id"] = le_name.fit_transform(h["name"].astype(str))

    hotel_stats = h.groupby("hotel_id").agg(
        place_enc        = ("place",   lambda x: le_place.fit_transform(x.astype(str))[0]),
        avg_price_per_day= ("price",   "mean"),
        avg_total_spend  = ("total",   "mean"),
        avg_days         = ("days",    "mean"),
        booking_volume   = ("travelCode","count"),
    ).reset_index()

    hotel_ids = hotel_stats["hotel_id"].values
    hotel_names_arr = le_name.inverse_transform(hotel_ids)

    feat_df = hotel_stats.drop(columns=["hotel_id"])
    scaler  = MinMaxScaler()
    matrix  = scaler.fit_transform(feat_df)

    return matrix, hotel_names_arr, le_name


# ─────────────────────────────────────────────────────────────
# 4.  Build booking lookup
# ─────────────────────────────────────────────────────────────
def build_booking_lookup(hotels):
    return (hotels.groupby("userCode")["name"]
            .apply(list).to_dict())


# ─────────────────────────────────────────────────────────────
# 5.  Train & save
# ─────────────────────────────────────────────────────────────
def train(model_dir: str = MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)

    users, hotels, flights = load_data()

    user_matrix, user_codes, user_feature_names = build_user_matrix(users, hotels, flights)
    hotel_matrix, hotel_names_arr, le_hotel     = build_hotel_matrix(hotels)
    booking_lookup                               = build_booking_lookup(hotels)

    # User–user similarity (collaborative)
    user_sim = cosine_similarity(user_matrix)

    # User–hotel similarity (content-based)
    # Align feature dims
    min_d = min(user_matrix.shape[1], hotel_matrix.shape[1])
    user_hotel_sim = cosine_similarity(user_matrix[:, :min_d],
                                       hotel_matrix[:, :min_d])

    # Hotel popularity (fallback / cold-start)
    all_booked  = [h for lst in booking_lookup.values() for h in lst]
    popularity  = dict(Counter(all_booked).most_common())

    artifacts = {
        "user_sim":         user_sim,
        "user_hotel_sim":   user_hotel_sim,
        "user_codes":       user_codes.tolist(),
        "hotel_names":      hotel_names_arr.tolist(),
        "booking_lookup":   {str(k): v for k, v in booking_lookup.items()},
        "popularity":       popularity,
        "hotel_profiles":   hotels.groupby("name").agg(
                                place          = ("place", "first"),
                                avg_price_day  = ("price", "mean"),
                                avg_total      = ("total", "mean"),
                                avg_days       = ("days",  "mean"),
                                bookings       = ("travelCode", "count"),
                            ).reset_index().to_dict(orient="records"),
    }

    joblib.dump(artifacts, os.path.join(model_dir, "recommendation_artifacts.pkl"))
    print(f"[INFO] Recommendation artifacts saved → {model_dir}")
    return artifacts


# ─────────────────────────────────────────────────────────────
# 6.  Inference helper  (importable from API / Streamlit)
# ─────────────────────────────────────────────────────────────
def recommend(user_code: int, artifacts: dict, top_k: int = 5) -> list:
    """
    Returns top_k hotel name strings for a given user_code.
    Hybrid: content-based score + collaborative boost.
    Cold-start fallback for unseen users.
    """
    user_codes  = artifacts["user_codes"]
    hotel_names = artifacts["hotel_names"]
    uh_sim      = artifacts["user_hotel_sim"]
    u_sim       = artifacts["user_sim"]
    booked      = artifacts["booking_lookup"].get(str(user_code), [])
    popularity  = artifacts["popularity"]

    if user_code not in user_codes:
        # Cold-start: return most popular unbooked hotels
        recs = [h for h in popularity if h not in booked]
        return recs[:top_k]

    idx = list(user_codes).index(user_code)

    # Content-based scores
    cb_scores = dict(zip(hotel_names, uh_sim[idx]))

    # Collaborative boost from top-5 similar users
    similar_users = np.argsort(-u_sim[idx])[1:6]
    collab_boost  = Counter()
    for su_idx in similar_users:
        su_code  = user_codes[su_idx]
        su_booked = artifacts["booking_lookup"].get(str(su_code), [])
        for h in su_booked:
            if h not in booked:
                collab_boost[h] += u_sim[idx][su_idx]

    # Hybrid score
    all_hotels = set(hotel_names)
    scores = {
        h: cb_scores.get(h, 0) + 0.3 * collab_boost.get(h, 0)
        for h in all_hotels
        if h not in booked
    }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked[:top_k]]


if __name__ == "__main__":
    train()
