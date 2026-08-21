# Turbine Predictive Maintenance — MLOps + Responsible AI + Agentic AI

## 1. Project Overview

This project is an end-to-end MLOps platform for predictive maintenance of aircraft turbine engines.

The machine-learning model predicts the **Remaining Useful Life (RUL)** of a turbine from sensor and operating-condition data.

The project is being developed in stages:

```text
Raw Turbine Data
       ↓
Data Processing
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Pusher
       ↓
FastAPI Prediction API
       ↓
HTML + CSS Dashboard
       ↓
Docker
       ↓
GitHub Actions CI/CD
       ↓
Kubernetes / Minikube
       ↓
Evidently AI Monitoring
       ↓
Responsible AI / XAI
       ↓
MongoDB
       ↓
Agentic AI
```

The final goal is to build a system that not only makes predictions, but also monitors data/model behavior, detects drift, supports responsible AI and explainability, and eventually uses Agentic AI to investigate problems and recommend actions.

---

# 2. Dataset

We use the **NASA C-MAPSS FD004 dataset**.

C-MAPSS stands for Commercial Modular Aero-Propulsion System Simulation.

The dataset simulates turbine/aircraft-engine degradation over time.

Think of it like this:

> An engine starts healthy, runs cycle after cycle, gradually degrades, and eventually approaches failure.

Our model learns the relationship between sensor measurements and remaining useful life.

## FD004 columns

The raw data contains:

- 1 engine/unit identifier
- 1 cycle number
- 3 operating settings
- 21 sensor measurements

Therefore:

```text
1 + 1 + 3 + 21 = 26 columns
```

For modeling/monitoring, identifier columns can be removed, leaving:

```text
24 usable columns
```

---

# 3. What RUL Means

RUL = **Remaining Useful Life**.

Example:

```text
Engine has 100 cycles left
        ↓
RUL = 100
```

Later:

```text
Engine has 50 cycles left
        ↓
RUL = 50
```

The goal is to predict degradation early enough to support predictive maintenance.

---

# 4. Project Structure

The project currently contains or uses components similar to:

```text
final-mlops/
│
├── artifact/
│   ├── model/
│   │   └── random_forest_model.pkl
│   ├── candidate_model/
│   │   └── random_forest_model.pkl
│   ├── production_model/
│   │   └── random_forest_model.pkl
│   └── processed/
│       └── scaler.pkl
│
├── config/
│   └── monitoring_config.yaml
│
├── data/
│   ├── raw/
│   │   └── cmapss/
│   │       ├── train_FD004.txt
│   │       └── test_FD004.txt
│   └── monitoring/
│       └── simulated_drift_FD004.csv
│
├── reports/
│   └── evidently/
│       ├── fd004_drift_report.html
│       ├── fd004_drift_results.json
│       └── monitoring_summary.json
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── monitoring/
│   │   ├── evidently_monitor.py
│   │   └── simulate_drift.py
│   └── pipeline/
│       └── training_pipeline.py
│
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The exact structure can grow as new components are added.

---

# 5. Stage 1 — Data Ingestion

The first stage brings FD004 into the project.

```text
NASA C-MAPSS FD004
        ↓
Data Ingestion
        ↓
Training/Test data
```

At the current stage, the data comes from local FD004 files.

### Future

MongoDB will eventually be added as a production/new-data source.

---

# 6. Stage 2 — Data Transformation

The transformation stage prepares raw data for machine learning.

Typical work includes:

- selecting useful columns
- handling data types
- separating features and target
- scaling/transforming data where required
- preparing training and validation data

The scaler is stored as:

```text
artifact/processed/scaler.pkl
```

The idea is:

```text
Raw data
   ↓
Transform
   ↓
ML-ready features
```

---

# 7. Stage 3 — Model Training

We trained a **Random Forest regression model** for RUL prediction.

The training output showed:

```text
Training data shape: (61249, 27)
Features used for training: 25
Training samples: 61249
```

The model was successfully trained and saved as:

```text
artifact/model/random_forest_model.pkl
```

---

# 8. Stage 4 — Model Evaluation

The trained model is evaluated against test data.

Important regression metrics:

- MAE
- RMSE
- R²

Our completed evaluation produced approximately:

```text
MAE  = 10.6008
RMSE = 18.0135
R²   = 0.8244
```

The evaluation also produced:

```text
Model approved: True
```

The flow is:

```text
Trained model
      ↓
Test data
      ↓
Prediction
      ↓
Metrics
      ↓
Approval decision
```

---

# 9. Stage 5 — Candidate Model

The candidate model represents the model being evaluated before production.

Path:

```text
artifact/candidate_model/random_forest_model.pkl
```

Flow:

```text
Newly trained model
        ↓
Candidate model
        ↓
Evaluation
        ↓
Approved?
```

---

# 10. Stage 6 — Model Pusher

The Model Pusher stage was completed successfully.

Its job:

```text
Candidate model
      ↓
Approved?
      ↓
YES
      ↓
Production model
```

Production model:

```text
artifact/production_model/random_forest_model.pkl
```

This separates training/evaluation from the model used by the application.

---

# 11. Stage 7 — MLflow

MLflow was integrated into the training pipeline for experiment/run tracking.

It helps track:

- experiment runs
- metrics
- parameters
- artifacts
- model-training history

Conceptually:

```text
Model training
      ↓
MLflow
      ↓
Metrics + parameters + artifacts
```

---

# 12. Stage 8 — FastAPI

FastAPI exposes the trained production model as an API.

Flow:

```text
User
 ↓
FastAPI
 ↓
Production Random Forest
 ↓
RUL prediction
 ↓
Response
```

The application runs on port:

```text
8000
```

A health endpoint was also added for Kubernetes health checks.

---

# 13. Stage 9 — HTML + CSS Dashboard

A frontend dashboard was created using:

```text
templates/index.html
static/style.css
```

The dashboard contains:

### Turbine Information

- Turbine ID
- Current Cycle

### Operating Settings

- Setting 1
- Setting 2
- Setting 3

### Sensor Data

The dashboard accepts turbine sensor measurements.

Flow:

```text
User enters turbine data
        ↓
Dashboard
        ↓
FastAPI
        ↓
ML model
        ↓
RUL prediction
        ↓
Dashboard result
```

The dashboard was successfully tested locally.

---

# 14. Stage 10 — Docker

The FastAPI application was containerized using Docker.

The Dockerfile installs dependencies, copies required project files, and starts Uvicorn.

Port:

```text
8000
```

Architecture:

```text
Project
  ↓
Dockerfile
  ↓
Docker image
  ↓
Container
  ↓
FastAPI
```

The Docker image was successfully built and run locally.

---

# 15. Stage 11 — GitHub Actions CI/CD

GitHub Actions was added to automate testing and Docker image publishing.

The pipeline includes steps such as:

```text
Checkout repository
        ↓
Setup Python
        ↓
Install dependencies
        ↓
Check Python files
        ↓
Test FastAPI import
        ↓
Build Docker image
        ↓
Push image
```

During development, CI issues involving model/scaler artifacts and dataset availability were fixed.

---

# 16. Git LFS

The production model is a binary `.pkl` file.

Git LFS was configured to track:

```text
artifact/production_model/random_forest_model.pkl
```

This helps manage ML binary artifacts without treating them like normal source code.

---

# 17. GitHub Container Registry

The Docker image is published to GitHub Container Registry (GHCR).

The image follows the pattern:

```text
ghcr.io/<github-username>/final-mlops:latest
```

The repository/package is private, so Kubernetes needs authentication to pull it.

---

# 18. Kubernetes

We introduced Kubernetes after Docker.

**Minikube** is being used as the local Kubernetes cluster.

Architecture:

```text
Kubernetes
   │
   ├── Deployment
   │      ↓
   │   FastAPI Pod
   │
   └── Service
          ↓
       Port 8000
```

The deployment runs the private GHCR image.

A Kubernetes image-pull secret was created:

```text
ghcr-secret
```

This allows Kubernetes to authenticate with GHCR.

---

# 19. Kubernetes Health Checks

The deployment uses:

### Liveness probe

Checks whether the application is alive.

### Readiness probe

Checks whether the application is ready to receive traffic.

Both use the FastAPI health endpoint:

```text
/health
```

Initially we had:

```text
ImagePullBackOff
```

because Kubernetes could not pull the private GHCR image.

After GHCR authentication was configured, the pod successfully reached:

```text
Running
```

The dashboard was then accessible through the Kubernetes service.

---

# 20. Kubernetes Service

A service similar to:

```text
turbine-mlops-service
```

exposes the FastAPI application inside Minikube.

The service forwards traffic to:

```text
8000
```

The Minikube service command can be used to access the application.

---

# 21. Evidently AI Monitoring

We installed:

```text
evidently==0.7.21
```

The first monitoring experiment compared:

```text
Reference:
FD004 training data

Current:
FD004 test data
```

The initial result showed no significant dataset drift.

That was expected and proved that the monitor can report a healthy condition.

---

# 22. Simulated Data Drift

To prove that Evidently can detect a problem, we created:

```text
data/monitoring/simulated_drift_FD004.csv
```

We deliberately changed:

```text
sensor_2
sensor_3
sensor_4
```

to simulate a production distribution shift.

This is a controlled test and is not intended to represent real physical turbine behavior.

---

# 23. Evidently Drift Result

Evidently successfully detected:

```text
Total columns: 24
Drifted columns: 3
Drift share: 12.50%
```

For example, `sensor_2` was detected as drifted.

The HTML report is:

```text
reports/evidently/fd004_drift_report.html
```

This proved that the monitoring system can identify changed sensor distributions.

---

# 24. Machine-Readable Monitoring Results

We created:

```text
reports/evidently/fd004_drift_results.json
```

This lets software consume the Evidently results without reading an HTML report.

We also created:

```text
reports/evidently/monitoring_summary.json
```

The summary contains simplified monitoring information such as:

```json
{
    "drifted_columns": 3,
    "drift_share_percentage": 12.5,
    "governance_status": "WARNING"
}
```

This file is especially useful for the future Agentic AI layer.

---

# 25. Monitoring Configuration

We created:

```text
config/monitoring_config.yaml
```

Current configuration:

```yaml
monitoring:
  drift_warning_threshold: 0.10
  drift_critical_threshold: 0.30
  reference_dataset: "FD004"
  report_format: "html"
  save_json: true
```

Our current governance policy is:

```text
0% – <10%     → HEALTHY
10% – <30%    → WARNING
30%+          → CRITICAL
```

Our simulated result:

```text
12.5%
   ↓
WARNING
```

The completed monitoring output was:

```text
Governance status: WARNING
Status: SUCCESS
```

---

# 26. Evidently vs Governance

These are two different things.

### Evidently

Evidently performs the statistical analysis.

It answers:

> Does this feature's data distribution look different?

### Our governance layer

Our configuration answers:

> How serious is the amount of detected drift for our system?

Therefore:

```text
Evidently
   ↓
Statistical result
   ↓
Governance policy
   ↓
HEALTHY / WARNING / CRITICAL
```

---

# 27. Current Architecture

```text
                NASA C-MAPSS FD004
                         │
                         ▼
                 Data Ingestion
                         │
                         ▼
                Data Transformation
                         │
                         ▼
                 Random Forest
                  RUL Prediction
                         │
                         ▼
                  Model Evaluation
                         │
                   Approved = True
                         │
                         ▼
                    Model Pusher
                         │
                         ▼
                 Production Model
                         │
                         ▼
                     FastAPI
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              HTML/CSS       REST API
                  │             │
                  └──────┬──────┘
                         ▼
                       Docker
                         │
                         ▼
                  GitHub Actions
                         │
                         ▼
                       GHCR
                         │
                         ▼
                    Kubernetes
                      Minikube
                         │
                         ▼
                    FastAPI Pod
                         │
                         ▼
                   User Dashboard


Monitoring:

Production / New Data
        │
        ▼
   Evidently AI
        │
        ▼
   Drift Detection
        │
        ▼
 Governance Config
        │
        ├── HEALTHY
        ├── WARNING
        └── CRITICAL
```

---

# 28. What Is Not Finished Yet

## A. MongoDB Data Ingestion

MongoDB will eventually replace the simulated current-data CSV.

Current:

```text
simulated_drift_FD004.csv
        ↓
Evidently
```

Future:

```text
MongoDB
   ↓
New turbine data
   ↓
Evidently
```

MongoDB ingestion was intentionally postponed.

---

## B. Responsible AI / Explainable AI

The next major area is Responsible AI.

Planned components include:

- Explainable AI
- feature importance
- prediction explanations
- model behavior monitoring
- governance
- appropriate bias/fairness checks where applicable

For this regression problem, explainability is especially useful.

The system should eventually answer:

> Why did the model predict this RUL?

Example:

```text
Predicted RUL: 59 cycles

Important contributing features:
sensor_11 → high influence
sensor_4  → medium influence
setting_2 → lower influence
```

---

# 29. Explainable AI Architecture

```text
Turbine sensor data
       ↓
Random Forest
       ↓
RUL prediction
       ↓
XAI
       ↓
Feature contribution
       ↓
Human-understandable explanation
```

The exact XAI library/technique will be selected when this stage is implemented.

---

# 30. Agentic AI

Agentic AI will be added later.

It should not simply be a chatbot placed on top of the project.

It should use actual outputs from the MLOps system.

Target architecture:

```text
Prediction
    +
Drift Detection
    +
Model Explanation
    +
Operational Information
    ↓
Agentic AI
    ↓
Reason
    ↓
Investigate
    ↓
Recommend action
```

Example:

```text
Evidently:
3 sensors drifted

XAI:
sensor_2 strongly influences the prediction

Agentic AI:
"Sensor 2 has significantly changed from the
reference distribution and strongly influences
RUL predictions. Investigate sensor 2 calibration
and collect additional production data."
```

---

# 31. Final Target Architecture

```text
                         ┌─────────────────┐
                         │   MongoDB       │
                         │ Production Data │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Data Monitoring │
                         │   Evidently     │
                         └────────┬────────┘
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                    No Drift              Drift
                       │                     │
                       │                     ▼
                       │              Governance Alert
                       │                     │
                       └──────────┬──────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Responsible AI  │
                         │      / XAI      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Agentic AI    │
                         │                 │
                         │ Investigate     │
                         │ Reason          │
                         │ Recommend       │
                         └────────┬────────┘
                                  │
                                  ▼
                         Human / Engineer
```

---

# 32. Technologies Used

| Area | Technology |
|---|---|
| Programming | Python |
| ML | Scikit-learn |
| Model | Random Forest Regressor |
| Dataset | NASA C-MAPSS FD004 |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Frontend | HTML + CSS |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry |
| Orchestration | Kubernetes |
| Local Kubernetes | Minikube |
| Monitoring | Evidently AI 0.7.21 |
| Configuration | YAML |
| Artifact Management | Git / Git LFS |
| Future Database | MongoDB |
| Future Explainability | XAI library |
| Future Intelligent Layer | Agentic AI |

---

# 33. Completed Milestones

- [x] FD004 dataset integrated
- [x] Data processing pipeline
- [x] Random Forest RUL model
- [x] Model training
- [x] Model evaluation
- [x] Model approval logic
- [x] Candidate model
- [x] Model pusher
- [x] Production model
- [x] MLflow experiment tracking
- [x] FastAPI API
- [x] HTML dashboard
- [x] CSS styling
- [x] Local prediction testing
- [x] Docker image
- [x] GitHub Actions CI/CD
- [x] Git LFS for model artifact
- [x] GHCR image publishing
- [x] Kubernetes deployment
- [x] Minikube deployment
- [x] Private GHCR authentication
- [x] Kubernetes service
- [x] Evidently AI monitoring
- [x] Simulated drift test
- [x] HTML drift report
- [x] JSON monitoring output
- [x] Monitoring summary
- [x] Monitoring configuration
- [x] Governance status calculation

---

# 34. Remaining Roadmap

```text
CURRENT
  │
  ├── MLOps foundation ✅
  ├── FastAPI ✅
  ├── Docker ✅
  ├── CI/CD ✅
  ├── Kubernetes ✅
  └── Evidently monitoring ✅
          │
          ▼
NEXT
  │
  ├── Responsible AI
  │     └── Explainable AI / XAI
  │
  ├── MongoDB ingestion
  │
  ├── Real production monitoring
  │
  ├── Automated monitoring/alerts
  │
  └── Agentic AI
        │
        ├── Read monitoring results
        ├── Investigate drift
        ├── Use XAI results
        └── Recommend actions
```

---

# 35. Project Goal in One Sentence

> Build an intelligent MLOps platform that predicts turbine Remaining Useful Life, continuously monitors data and model behavior, detects drift, supports responsible and explainable AI, and eventually uses Agentic AI to investigate issues and recommend maintenance actions.

---

# 36. Simple Explanation of the Whole Project

If explaining the project to someone who knows nothing about it:

> We have simulated aircraft turbine data. We train a machine-learning model to predict how much useful life the turbine has left. We package the model into an API and deploy it using Docker and Kubernetes. We then monitor incoming data using Evidently AI to detect when sensor behavior changes. Governance rules classify the severity of that change. Next, we will add explainability and Responsible AI so engineers can understand why the model made a prediction. Finally, an Agentic AI system will use all this information to investigate problems and recommend actions.

---

# 37. Current Status

| Component | Status |
|---|---|
| Dataset / Data Pipeline | ✅ Complete |
| Model Training | ✅ Complete |
| Model Evaluation | ✅ Complete |
| Candidate Model | ✅ Complete |
| Model Pusher | ✅ Complete |
| Production Model | ✅ Complete |
| MLflow | ✅ Complete |
| FastAPI | ✅ Complete |
| HTML/CSS Dashboard | ✅ Complete |
| Docker | ✅ Complete |
| GitHub Actions CI/CD | ✅ Complete |
| GHCR | ✅ Complete |
| Kubernetes / Minikube | ✅ Complete |
| Evidently Monitoring | ✅ Complete |
| Drift Simulation | ✅ Complete |
| Governance Configuration | ✅ Complete |
| MongoDB Ingestion | ⏳ Planned |
| Responsible AI / XAI | ⏳ Next |
| Agentic AI | ⏳ Later |

---

## Final Takeaway

Do not think of this project as just:

```text
Random Forest + FastAPI
```

The stronger version is:

```text
ML
+
MLOps
+
Deployment
+
Monitoring
+
Governance
+
Responsible AI
+
Agentic AI
```

That combination is the main value of the project.
