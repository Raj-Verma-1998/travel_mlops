# ✈️ Travel MLOps Capstone Project

> End-to-end Machine Learning + MLOps platform for travel analytics —  
> regression, classification, recommendations, REST API, Docker, Kubernetes, Airflow, Jenkins & MLflow.

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Datasets](#3-datasets)
4. [Models](#4-models)
5. [REST API](#5-rest-api)
6. [Containerisation (Docker)](#6-containerisation-docker)
7. [Kubernetes Deployment](#7-kubernetes-deployment)
8. [Apache Airflow Pipelines](#8-apache-airflow-pipelines)
9. [CI/CD with Jenkins](#9-cicd-with-jenkins)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Streamlit Dashboard](#11-streamlit-dashboard)
12. [Testing](#12-testing)
13. [Quick Start](#13-quick-start)
14. [Results Summary](#14-results-summary)

---

## 1. Project Overview

This project demonstrates a full **MLOps lifecycle** applied to a travel & tourism dataset.  
Three ML models are built, tracked, served, containerised, and deployed via automated pipelines:

| Objective | Implementation |
|---|---|
| Flight price prediction | XGBoost Regression + MLflow |
| User gender classification | GradientBoosting Classifier + MLflow |
| Hotel recommendations | Hybrid content-based + collaborative filtering |
| REST API | Flask + Gunicorn |
| Containerisation | Docker multi-stage build |
| Orchestration | Kubernetes (HPA, rolling updates) |
| Workflow automation | Apache Airflow DAG |
| CI/CD | Jenkins pipeline |
| Experiment tracking | MLflow tracking server |
| Dashboard | Streamlit web application |

---

## 2. Repository Structure

```
travel_mlops/
├── data/
│   ├── flights.csv          # 271,888 flight records
│   ├── hotels.csv           # 40,552 hotel bookings
│   └── users.csv            # 1,340 user profiles
│
├── models/
│   ├── train_regression.py      # XGBoost flight price model
│   ├── train_classification.py  # Gender classification model
│   ├── train_recommendation.py  # Hotel recommendation engine
│   └── artifacts/               # Saved model .pkl files + metrics
│
├── api/
│   └── app.py               # Flask REST API (4 endpoints)
│
├── docker/
│   ├── Dockerfile            # Multi-stage production image
│   └── docker-compose.yml    # Full stack: API + MLflow + Airflow + PostgreSQL
│
├── kubernetes/
│   ├── travel-api-deployment.yaml  # API Deployment, Service, Ingress, HPA
│   └── mlflow-deployment.yaml      # MLflow server deployment
│
├── airflow/
│   └── dags/
│       └── travel_ml_pipeline.py   # Full ML pipeline DAG (@weekly)
│
├── jenkins/
│   └── Jenkinsfile           # CI/CD pipeline (lint → train → build → deploy)
│
├── mlflow_config/            # MLflow server configuration
├── streamlit/
│   └── app.py                # Interactive analytics dashboard
│
├── tests/
│   └── test_api.py           # 26 unit tests (100% passing)
│
├── requirements.txt
└── README.md
```

---

## 3. Datasets

### Flights (`flights.csv`)
- **271,888 rows** | 10 columns
- Origin/destination: Brazilian cities
- 3 flight classes: `firstClass`, `premium`, `economic`
- 3 agencies: `FlyingDrops`, `CloudFy`, `Rainbow`
- Price range: $301 – $1,754

### Hotels (`hotels.csv`)
- **40,552 rows** | 8 columns
- 9 unique hotels across 9 Brazilian destinations
- Price per day + total cost

### Users (`users.csv`)
- **1,340 rows** | 5 columns
- Gender: `male` / `female` / `none` (near-equal split)
- Age: 21–65 | 5 companies

---

## 4. Models

### 4.1 Flight Price Regression
**Algorithm:** XGBoost Regressor  
**Features:** `flightType`, `agency`, `from/to` (encoded), `time`, `distance`, `speed`, `price_per_km`, `month`, `dayofweek`, `year`  
**MLflow experiment:** `flight_price_regression`

| Metric | Value |
|--------|-------|
| R²     | 1.0000 |
| MAE    | 1.23  |
| RMSE   | 1.65  |
| MAPE   | 0.14% |

> The extremely high R² reflects that the dataset's price is a near-deterministic function of `flightType × distance × time`, which is typical of synthetic/generated travel datasets.

**Top features:**
1. `time` (0.296)
2. `distance` (0.253)
3. `flightType` (0.220)
4. `price_per_km` (0.166)

---

### 4.2 Gender Classification
**Algorithm:** GradientBoosting Classifier  
**Features:** Age, company, per-user flight & hotel aggregates (19 features total)  
**MLflow experiment:** `gender_classification`

| Metric | Value |
|--------|-------|
| Accuracy    | 0.3321 |
| ROC-AUC     | 0.5032 |
| CV Mean     | 0.2843 |
| CV Std      | ±0.0397 |

> Accuracy close to random chance (33.3% = 1/3 for 3 classes) confirms that travel behaviour is **not predictive of gender** in this dataset — the labels are effectively randomly distributed. This is the correct and honest result; a model reporting 90%+ accuracy here would be overfitting.

---

### 4.3 Hotel Recommendation
**Algorithm:** Hybrid content-based + collaborative filtering  
**Method:**
1. Build user feature vectors (flight + hotel aggregates)
2. Build hotel feature vectors (price, stay length, location)
3. Compute user–hotel cosine similarity (content-based)
4. Collaborative boost from top-5 similar users' bookings
5. Cold-start fallback to global popularity ranking

**Hybrid score:**
```
score(hotel) = cosine_sim(user, hotel) + 0.3 × Σ sim(user, neighbour) × [hotel ∈ neighbour.bookings]
```

---

## 5. REST API

**Base URL:** `http://localhost:5000`

### Endpoints

#### `GET /health`
Returns service status and loaded model inventory.
```json
{
  "status": "ready",
  "models_loaded": {"flight_model": true, "gender_model": true, "rec_artifacts": true}
}
```

#### `POST /predict/flight-price`
```json
// Request
{
  "from": "Recife (PE)", "to": "Florianopolis (SC)",
  "flightType": "firstClass", "agency": "FlyingDrops",
  "time": 1.76, "distance": 676.53,
  "year": 2024, "month": 6, "dayofweek": 2
}

// Response
{ "predicted_price": 1415.22, "currency": "USD", "model": "XGBoost Regressor" }
```

#### `POST /predict/gender`
```json
// Request – pass any user feature columns (missing → 0)
{ "age": 35, "flight_count": 12, "avg_flight_price": 950.0, ... }

// Response
{
  "predicted_gender": "female",
  "confidence": 0.4821,
  "probabilities": {"female": 0.4821, "male": 0.3102, "none": 0.2077}
}
```

#### `POST /recommend/hotels`
```json
// Request
{ "user_code": 42, "top_k": 5 }

// Response
{
  "user_code": 42,
  "recommendations": [
    {"rank": 1, "hotel": "Hotel G", "avg_price_day": 287.5,
     "avg_total": 1150.0, "avg_days": 4.0, "place": "Maceio (AL)"},
    ...
  ]
}
```

---

## 6. Containerisation (Docker)

### Build & run locally
```bash
# Build image
docker build -f docker/Dockerfile -t travel-mlops/travel-api:latest .

# Run single container
docker run -p 5000:5000 \
  -v $(pwd)/models/artifacts:/app/models/artifacts \
  travel-mlops/travel-api:latest

# Full stack (API + MLflow + Airflow + PostgreSQL)
cd docker
docker-compose up -d

# Service URLs
# API:     http://localhost:5000
# MLflow:  http://localhost:5001
# Airflow: http://localhost:8080  (admin / admin)
```

### Image design
- Multi-stage build (builder → slim runtime)
- Non-root `appuser` for security
- Gunicorn with 2 workers
- Docker `HEALTHCHECK` on `/health`

---

## 7. Kubernetes Deployment

```bash
# Create namespace and deploy all resources
kubectl apply -f kubernetes/

# Check rollout
kubectl rollout status deployment/travel-api -n travel-mlops

# Scale manually
kubectl scale deployment travel-api --replicas=5 -n travel-mlops

# View HPA status
kubectl get hpa -n travel-mlops
```

### Key manifests
| Resource | Config |
|---|---|
| Deployment | 3 replicas, rolling update (0 downtime) |
| HPA | Min 2 → Max 10 pods; CPU 60% / Memory 75% |
| Service | ClusterIP on port 80 |
| Ingress | `travel-api.local` via nginx |
| PVC | 2Gi ReadOnlyMany for model artifacts |

---

## 8. Apache Airflow Pipelines

**DAG:** `travel_ml_pipeline`  
**Schedule:** `@weekly` (Sunday 00:00 UTC)

### DAG Graph
```
start
  └─ data_validation
       └─ feature_engineering
            ├─ train_regression    ─┐
            ├─ train_classification ├─ evaluate_models → register_models → notify → end
            └─ train_recommendation─┘
```

### Tasks
| Task | Description |
|---|---|
| `data_validation` | Assert CSVs exist with correct columns |
| `feature_engineering` | Compute derived features, save `flights_engineered.csv` |
| `train_regression` | Retrain XGBoost, push metrics via XCom |
| `train_classification` | Retrain GradientBoosting, push metrics |
| `train_recommendation` | Rebuild cosine similarity artifacts |
| `evaluate_models` | Assert R² ≥ 0.85, Accuracy ≥ 0.30 |
| `register_models` | Write `model_registry.json` manifest |
| `notify` | Print deployment summary |

### Start Airflow locally
```bash
# Using docker-compose (recommended)
cd docker && docker-compose up airflow-web airflow-scheduler -d

# Or standalone
pip install apache-airflow
airflow db init
airflow webserver -p 8080 &
airflow scheduler &
# Copy DAG: cp airflow/dags/travel_ml_pipeline.py ~/airflow/dags/
```

---

## 9. CI/CD with Jenkins

**File:** `jenkins/Jenkinsfile`

### Pipeline stages
```
Checkout → Lint & Test → Train Models (parallel) → Evaluate → Build Image → Push Image → Deploy (K8s) → Smoke Test → Notify
```

| Stage | Action |
|---|---|
| Lint & Test | flake8 + pytest with coverage |
| Train Models | 3 parallel training stages |
| Evaluate | Assert metric thresholds; fail-fast |
| Build Image | `docker build` with build labels |
| Push Image | Push `:BUILD_NUMBER` and `:latest` tags |
| Deploy | `kubectl set image` + rollout status |
| Smoke Test | `curl /health` against live service |

### Setup
1. Install plugins: `Docker Pipeline`, `Kubernetes CLI`, `Pipeline`
2. Add credentials: `docker-hub-creds`, `kubeconfig-prod`
3. Create Pipeline job → point to `jenkins/Jenkinsfile`
4. Trigger on push to `main` or `develop`

---

## 10. MLflow Experiment Tracking

Two experiments are created automatically during training:

| Experiment | Runs | Tracked |
|---|---|---|
| `flight_price_regression` | XGBoost runs | params, MAE, RMSE, R², MAPE, model artifact |
| `gender_classification` | GradientBoosting runs | params, accuracy, ROC-AUC, CV scores, model artifact |

### Access MLflow UI
```bash
# Start server
mlflow ui --port 5001

# Or via Docker Compose
docker-compose up mlflow-server
# → http://localhost:5001
```

### Key MLflow features used
- `mlflow.set_experiment()` — named experiment grouping
- `mlflow.log_params()` — hyperparameter logging
- `mlflow.log_metrics()` — evaluation metric logging
- `mlflow.xgboost.log_model()` — model serialisation
- `mlflow.sklearn.log_model()` — sklearn model serialisation

---

## 11. Streamlit Dashboard

```bash
cd travel_mlops
streamlit run streamlit/app.py
# → http://localhost:8501
```

### Pages
| Page | Features |
|---|---|
| 📊 EDA & Insights | Price distributions, agency comparison, monthly trends, hotel popularity, user demographics |
| 💰 Flight Price Predictor | Interactive form → real-time price prediction + historical comparables |
| 👤 Gender Classifier | User lookup → prediction + probability bar chart |
| 🏨 Hotel Recommender | User code → ranked hotel cards with place, price, and booking history |

---

## 12. Testing

```bash
# Run all 26 tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=api --cov=models --cov-report=term-missing
```

### Test coverage

| Test class | Tests | Coverage |
|---|---|---|
| `TestHealth` | 2 | `/health` endpoint |
| `TestFlightPrice` | 5 | Prediction correctness, range, edge cases |
| `TestGenderClassification` | 5 | Output validity, probability sum |
| `TestHotelRecommendation` | 5 | Counts, cold-start, structure |
| `TestModelArtifacts` | 5 | File existence, metric thresholds |
| `TestDataQuality` | 4 | Null checks, value ranges, consistency |
| **Total** | **26** | **100% passing** |

---

## 13. Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd travel_mlops
pip install -r requirements.txt

# 2. Train all models
python models/train_regression.py
python models/train_classification.py
python models/train_recommendation.py

# 3. Run tests
pytest tests/ -v

# 4. Start API
python api/app.py
# → http://localhost:5000/health

# 5. Launch dashboard
streamlit run streamlit/app.py
# → http://localhost:8501

# 6. Full Docker stack
cd docker && docker-compose up -d
```

### API test commands
```bash
# Health
curl http://localhost:5000/health

# Flight price
curl -X POST http://localhost:5000/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops","time":1.76,"distance":676.53,"year":2019,"month":9,"dayofweek":3}'

# Gender
curl -X POST http://localhost:5000/predict/gender \
  -H "Content-Type: application/json" \
  -d '{"age":35,"flight_count":12,"avg_flight_price":950}'

# Recommendations
curl -X POST http://localhost:5000/recommend/hotels \
  -H "Content-Type: application/json" \
  -d '{"user_code":42,"top_k":5}'
```

---

## 14. Results Summary

| Component | Status | Notes |
|---|---|---|
| Regression model | ✅ R²=1.00, RMSE=1.65 | Price is deterministic in dataset |
| Classification model | ✅ Acc=33.2% | Gender is random w.r.t. travel behaviour |
| Recommendation model | ✅ Deployed | Hybrid CB+CF with cold-start |
| REST API | ✅ 4 endpoints | Flask + Gunicorn |
| Docker | ✅ Multi-stage | Non-root, healthcheck |
| Docker Compose | ✅ 5 services | API, MLflow, Airflow, PostgreSQL |
| Kubernetes | ✅ Full manifests | HPA, rolling deploy, ingress |
| Airflow DAG | ✅ @weekly | 8 tasks, parallel training |
| Jenkins CI/CD | ✅ Full pipeline | Lint→Train→Build→Deploy→Smoke |
| MLflow tracking | ✅ 2 experiments | Params, metrics, model artifacts |
| Streamlit dashboard | ✅ 4 pages | EDA, predictor, classifier, recommender |
| Unit tests | ✅ 26/26 passed | API, models, data quality |
