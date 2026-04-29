"""
Travel Analytics & Recommendation Dashboard
============================================
Streamlit web application providing:
  • EDA visualisations across flights, hotels, and users
  • Live flight-price predictions
  • Gender classification
  • Personalised hotel recommendations
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models", "artifacts")
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Travel ML Analytics",
    page_icon   = "✈️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:700; color:#1a1a2e; }
    .subtitle    { font-size:1.1rem; color:#666; margin-bottom:1.5rem; }
    .metric-card { background:#f0f4ff; border-radius:12px;
                   padding:1rem; text-align:center; }
    .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; }
    div[data-testid="metric-container"] {
        background-color: #f0f4ff;
        border: 1px solid #d0d8ff;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Data & model cache
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    users   = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    flights["date"] = pd.to_datetime(flights["date"], dayfirst=False, errors="coerce")
    hotels["date"]  = pd.to_datetime(hotels["date"],  dayfirst=False, errors="coerce")
    flights["month"] = flights["date"].dt.month
    flights["year"]  = flights["date"].dt.year
    return flights, hotels, users

@st.cache_resource(show_spinner=False)
def load_models():
    arts = {}
    def _load(k, p):
        try:    arts[k] = joblib.load(p)
        except: pass
    def _json(k, p):
        try:
            with open(p) as f: arts[k] = json.load(f)
        except: pass

    _load("flight_model",    os.path.join(MODEL_DIR, "flight_price_model.pkl"))
    _load("flight_scaler",   os.path.join(MODEL_DIR, "flight_scaler.pkl"))
    _load("flight_encoders", os.path.join(MODEL_DIR, "flight_encoders.pkl"))
    _json("flight_features", os.path.join(MODEL_DIR, "flight_feature_names.json"))

    _load("gender_model",    os.path.join(MODEL_DIR, "gender_model.pkl"))
    _load("gender_scaler",   os.path.join(MODEL_DIR, "gender_scaler.pkl"))
    _load("gender_encoder",  os.path.join(MODEL_DIR, "gender_encoder.pkl"))
    _json("gender_features", os.path.join(MODEL_DIR, "gender_feature_names.json"))

    _load("rec_artifacts",   os.path.join(MODEL_DIR, "recommendation_artifacts.pkl"))
    return arts

flights, hotels, users = load_data()
arts = load_models()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-take-off.png", width=80)
    st.markdown("## ✈️ Travel ML Ops")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 EDA & Insights",
         "💰 Flight Price Predictor",
         "👤 Gender Classifier",
         "🏨 Hotel Recommender"],
        index=0
    )
    st.markdown("---")

    # Quick KPIs
    st.markdown("### Dataset Stats")
    st.metric("Total Flights",  f"{len(flights):,}")
    st.metric("Total Bookings", f"{len(hotels):,}")
    st.metric("Total Users",    f"{len(users):,}")


# ═════════════════════════════════════════════════════════════
# PAGE 1 – EDA & Insights
# ═════════════════════════════════════════════════════════════
if page == "📊 EDA & Insights":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep dive into travel patterns, pricing, and user behaviour</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✈️ Flights", "🏨 Hotels", "👥 Users"])

    # ── Flights tab ───────────────────────────────────────────
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Price",    f"${flights['price'].mean():,.0f}")
        col2.metric("Avg Distance", f"{flights['distance'].mean():,.0f} km")
        col3.metric("Avg Duration", f"{flights['time'].mean():.1f} hrs")
        col4.metric("Routes",       f"{flights[['from','to']].drop_duplicates().shape[0]:,}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                flights, x="price", nbins=60,
                color="flightType",
                title="Flight Price Distribution by Class",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"price": "Price (USD)", "flightType": "Class"},
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.box(
                flights, x="flightType", y="price",
                color="flightType",
                title="Price by Flight Type",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            agency_avg = (flights.groupby("agency")["price"]
                          .mean().reset_index()
                          .rename(columns={"price": "avg_price"}))
            fig = px.bar(
                agency_avg, x="agency", y="avg_price",
                color="agency", title="Average Price by Agency",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.scatter(
                flights.sample(min(5000, len(flights))),
                x="distance", y="price", color="flightType",
                opacity=0.5,
                title="Price vs Distance",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Monthly trend
        monthly = (flights.groupby(["year","month"])["price"]
                   .mean().reset_index())
        monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
        fig = px.line(
            monthly.sort_values("period"),
            x="period", y="price",
            title="Monthly Average Flight Price Trend",
            markers=True,
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # ── Hotels tab ────────────────────────────────────────────
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Stay",        f"{hotels['days'].mean():.1f} days")
        col2.metric("Avg Daily Rate",  f"${hotels['price'].mean():,.0f}")
        col3.metric("Avg Total Spend", f"${hotels['total'].mean():,.0f}")
        col4.metric("Unique Hotels",   f"{hotels['name'].nunique()}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            hotel_pop = (hotels.groupby("name")
                         .size().reset_index(name="bookings")
                         .sort_values("bookings", ascending=False))
            fig = px.bar(
                hotel_pop, x="name", y="bookings",
                color="bookings", title="Hotel Booking Popularity",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.box(
                hotels, x="name", y="total",
                title="Total Spend Distribution by Hotel",
                color="name",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        place_avg = (hotels.groupby("place")
                     .agg(avg_price=("price","mean"),
                          bookings=("travelCode","count"))
                     .reset_index())
        fig = px.scatter(
            place_avg, x="bookings", y="avg_price",
            size="bookings", text="place",
            title="Destination: Bookings vs Average Price",
            size_max=50,
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # ── Users tab ─────────────────────────────────────────────
    with tab3:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users",  len(users))
        col2.metric("Avg Age",      f"{users['age'].mean():.0f} yrs")
        col3.metric("Companies",    users["company"].nunique())

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                users, names="gender",
                title="Gender Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                users, x="age", color="gender",
                nbins=25,
                title="Age Distribution by Gender",
                barmode="overlay", opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            users, x="company", y="age",
            color="company",
            title="Age Distribution by Company",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 – Flight Price Predictor
# ═════════════════════════════════════════════════════════════
elif page == "💰 Flight Price Predictor":
    st.markdown('<p class="main-title">💰 Flight Price Predictor</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict flight cost using the trained XGBoost model</p>',
                unsafe_allow_html=True)

    if "flight_model" not in arts:
        st.error("⚠️ Flight price model not found. Run `python models/train_regression.py` first.")
        st.stop()

    all_from    = sorted(flights["from"].unique())
    all_to      = sorted(flights["to"].unique())
    all_types   = sorted(flights["flightType"].unique())
    all_agencies= sorted(flights["agency"].unique())

    with st.form("flight_form"):
        c1, c2 = st.columns(2)
        origin  = c1.selectbox("Origin",      all_from)
        dest    = c2.selectbox("Destination", all_to)

        c3, c4 = st.columns(2)
        ftype   = c3.selectbox("Flight Class",  all_types)
        agency  = c4.selectbox("Agency",        all_agencies)

        c5, c6 = st.columns(2)
        time_h  = c5.slider("Flight Duration (hrs)", 0.5, 12.0, 2.5, 0.25)
        dist_km = c6.slider("Distance (km)",          100, 5000, 700, 50)

        c7, c8, c9 = st.columns(3)
        year   = c7.selectbox("Year",       [2019, 2020, 2021, 2022, 2023, 2024])
        month  = c8.slider("Month",         1, 12, 6)
        dow    = c9.slider("Day of Week",   0, 6, 2)

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        model    = arts["flight_model"]
        scaler   = arts["flight_scaler"]
        encoders = arts["flight_encoders"]
        features = arts["flight_features"]

        row = {}
        for col, val in [("from", origin), ("to", dest),
                          ("flightType", ftype), ("agency", agency)]:
            le = encoders.get(col)
            row[col] = int(le.transform([val])[0]) if le and val in le.classes_ else 0

        row["time"]        = time_h
        row["distance"]    = dist_km
        row["year"]        = year
        row["month"]       = month
        row["dayofweek"]   = dow
        row["speed"]       = dist_km / (time_h + 1e-9)
        row["price_per_km"]= 0.0

        X_df = pd.DataFrame([row])[features]
        X_sc = scaler.transform(X_df)
        price = float(model.predict(X_sc)[0])

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.success(f"### Predicted Price: **${price:,.2f} USD**")
            st.info(f"Route: **{origin}** → **{dest}** | Class: **{ftype}** | Agency: **{agency}**")

        # Show similar prices from dataset
        st.markdown("#### 📊 Comparable Historical Flights")
        mask = (
            (flights["from"] == origin) &
            (flights["to"]   == dest)   &
            (flights["flightType"] == ftype)
        )
        similar = flights[mask][["from","to","flightType","agency","price","distance","time"]]
        if not similar.empty:
            st.dataframe(similar.sample(min(10, len(similar))).reset_index(drop=True),
                         use_container_width=True)
        else:
            st.info("No historical data for this exact route/class combination.")


# ═════════════════════════════════════════════════════════════
# PAGE 3 – Gender Classifier
# ═════════════════════════════════════════════════════════════
elif page == "👤 Gender Classifier":
    st.markdown('<p class="main-title">👤 Gender Classifier</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict user gender from travel behaviour features</p>',
                unsafe_allow_html=True)

    if "gender_model" not in arts:
        st.error("⚠️ Gender model not found. Run `python models/train_classification.py` first.")
        st.stop()

    features = arts["gender_features"]

    # Show model metrics
    metrics_path = os.path.join(MODEL_DIR, "gender_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            gm = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",   f"{gm.get('accuracy', 'N/A')}")
        c2.metric("ROC-AUC",    f"{gm.get('roc_auc',  'N/A')}")
        c3.metric("CV Mean",    f"{gm.get('cv_mean',  'N/A')}")
        c4.metric("CV Std",     f"±{gm.get('cv_std',  'N/A')}")

    st.markdown("---")
    st.markdown("#### Enter user travel profile:")

    # Smart input – look up existing user
    user_code = st.number_input("Lookup existing user code (0–1339)", 0, 1339, 42)

    # Merge to get features for the selected user
    f_agg = flights.groupby("userCode").agg(
        flight_count        = ("travelCode","count"),
        avg_flight_price    = ("price","mean"),
        total_flight_spend  = ("price","sum"),
        avg_flight_time     = ("time","mean"),
        avg_distance        = ("distance","mean"),
        max_flight_price    = ("price","max"),
        min_flight_price    = ("price","min"),
    ).reset_index()
    ftype_d = (flights.groupby(["userCode","flightType"]).size()
               .unstack(fill_value=0)
               .rename(columns=lambda c: f"flights_{c}"))
    agency_d = (flights.groupby(["userCode","agency"]).size()
                .unstack(fill_value=0)
                .rename(columns=lambda c: f"agency_{c}"))
    h_agg = hotels.groupby("userCode").agg(
        hotel_count         = ("travelCode","count"),
        avg_hotel_days      = ("days","mean"),
        avg_hotel_price_day = ("price","mean"),
        avg_hotel_total     = ("total","mean"),
        total_hotel_spend   = ("total","sum"),
    ).reset_index()

    from sklearn.preprocessing import LabelEncoder
    u_merged = users.merge(f_agg,   left_on="code", right_on="userCode", how="left")
    u_merged = u_merged.merge(ftype_d,  left_on="code", right_on="userCode", how="left")
    u_merged = u_merged.merge(agency_d, left_on="code", right_on="userCode", how="left")
    u_merged = u_merged.merge(h_agg,    left_on="code", right_on="userCode", how="left")
    u_merged.fillna(0, inplace=True)
    le_c = LabelEncoder(); u_merged["company_enc"] = le_c.fit_transform(u_merged["company"].astype(str))

    user_row = u_merged[u_merged["code"] == user_code]
    if user_row.empty:
        st.warning("User code not found.")
        st.stop()

    actual_gender = user_row["gender"].values[0]

    row_dict = {f: float(user_row[f].values[0]) if f in user_row.columns else 0.0
                for f in features}
    X_df = pd.DataFrame([row_dict])[features]
    X_sc = arts["gender_scaler"].transform(X_df)
    pred_idx   = arts["gender_model"].predict(X_sc)[0]
    pred_prob  = arts["gender_model"].predict_proba(X_sc)[0]
    le_g       = arts["gender_encoder"]
    pred_label = le_g.inverse_transform([pred_idx])[0]
    confidence = float(max(pred_prob))

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Gender", pred_label.capitalize())
    col2.metric("Confidence",       f"{confidence:.1%}")
    col3.metric("Actual Gender",    actual_gender.capitalize())

    # Probability bar chart
    prob_df = pd.DataFrame({
        "Gender":      le_g.classes_,
        "Probability": pred_prob,
    })
    fig = px.bar(
        prob_df, x="Gender", y="Probability",
        color="Gender", title="Prediction Probabilities",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=prob_df["Probability"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 – Hotel Recommender
# ═════════════════════════════════════════════════════════════
elif page == "🏨 Hotel Recommender":
    st.markdown('<p class="main-title">🏨 Hotel Recommender</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Personalised hotel suggestions powered by hybrid collaborative filtering</p>',
                unsafe_allow_html=True)

    if "rec_artifacts" not in arts:
        st.error("⚠️ Recommendation model not found. Run `python models/train_recommendation.py` first.")
        st.stop()

    rec_arts  = arts["rec_artifacts"]
    profiles  = {h["name"]: h for h in rec_arts.get("hotel_profiles", [])}

    col1, col2 = st.columns([2, 1])
    user_code = col1.number_input("User Code", 0, 1339, 42)
    top_k     = col2.slider("Recommendations", 1, 9, 5)

    if st.button("🔍 Get Recommendations", use_container_width=True):
        from models.train_recommendation import recommend
        recs = recommend(user_code, rec_arts, top_k=top_k)

        if not recs:
            st.warning("No recommendations found for this user.")
        else:
            st.markdown(f"### 🏆 Top {len(recs)} Hotels for User {user_code}")

            # Show as cards
            cols = st.columns(min(3, len(recs)))
            for i, hotel in enumerate(recs):
                with cols[i % 3]:
                    p = profiles.get(hotel, {})
                    st.markdown(f"""
                    <div style="background:#f8f9ff;border-radius:12px;
                                padding:1rem;margin-bottom:0.8rem;
                                border-left:4px solid #4361ee;">
                        <h4 style="color:#1a1a2e;margin:0">🏨 {hotel}</h4>
                        <p style="color:#555;margin:4px 0">📍 {p.get('place','N/A')}</p>
                        <p style="color:#555;margin:4px 0">💰 ${p.get('avg_price_day',0):.0f}/night</p>
                        <p style="color:#555;margin:4px 0">📅 Avg stay: {p.get('avg_days',0):.1f} days</p>
                        <p style="color:#555;margin:4px 0">🧾 Avg total: ${p.get('avg_total',0):.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Show past bookings
            st.markdown("#### 📋 User's Booking History")
            booked = rec_arts["booking_lookup"].get(str(user_code), [])
            if booked:
                hist = hotels[hotels["userCode"] == user_code][
                    ["name","place","days","price","total","date"]
                ].sort_values("date", ascending=False)
                st.dataframe(hist.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No previous bookings found (cold-start user).")

            # Popularity chart
            st.markdown("#### 📊 Recommended Hotels – Popularity vs Avg Price")
            rec_data = [
                {"Hotel": h,
                 "Bookings": profiles[h]["bookings"]    if h in profiles else 0,
                 "Avg Price": profiles[h]["avg_price_day"] if h in profiles else 0}
                for h in recs
            ]
            rec_df = pd.DataFrame(rec_data)
            fig = px.scatter(
                rec_df, x="Bookings", y="Avg Price",
                text="Hotel", size="Bookings",
                color="Hotel", size_max=40,
                title="Recommended Hotels: Popularity vs Price",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
