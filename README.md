# Brahma — The Creator Intelligence

> *"Tell me your goal and your data source. Nothing else is required."*

Brahma is an autonomous ML super-agent built on top of Claude Code. You describe a business problem in plain English, point it at your data, and Brahma runs the full machine learning pipeline — from raw data ingestion to a deployed, validated model — without requiring you to write a single line of code.

---

## Table of Contents

- [What is Brahma?](#what-is-brahma)
- [How to Activate](#how-to-activate)
- [Pipeline Stages](#pipeline-stages)
- [Supported Problem Types](#supported-problem-types)
- [Supported Data Sources](#supported-data-sources)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Skills](#skills)
- [Outputs](#outputs)
- [Example Session](#example-session)
- [Error Handling](#error-handling)

---

## What is Brahma?

Brahma is a **Claude Code super-agent** — a system of specialised agents and skills orchestrated by a single routing brain. It handles:

- Automatic problem classification (classification, regression, clustering, semi-supervised)
- Data ingestion from any source (files, databases, cloud, APIs)
- End-to-end ML pipeline execution across 9 stages
- Hyperparameter tuning, ensembling, cross-validation, and SHAP explainability
- UAT testing and deployment packaging
- A live dashboard for monitoring pipeline progress

Brahma is opinionated about quality. Every output is presentable to a CXO on Monday morning.

---

## How to Activate

Open Claude Code in this directory and say:

```
Wake Up Brahma
```

Brahma will greet you, then ask for two things:

| Input | Description | Example |
|-------|-------------|---------|
| **Goal** | What you want to achieve, in plain English | `"Predict which customers will churn next month"` |
| **Data Source** | Where your data lives | `data/customers.csv` or `postgresql://user:pass@host/db` |

Brahma will echo back its understanding and **wait for your confirmation** before running a single line of code.

---

## Pipeline Stages

Once confirmed, Brahma executes the following stages in order:

| Stage | Script | Description |
|-------|--------|-------------|
| **Stage 3** | `stage3_eda.py` | Exploratory Data Analysis — distributions, correlations, target analysis |
| **Stage 4** | `stage4_features.py` | Feature Engineering — encoding, scaling, selection, new feature creation |
| **Stage 6** | `stage6_train.py` | Model Training — baseline + XGBoost with Optuna hyperparameter tuning |
| **Stage 7** | `stage7_evaluate.py` | Model Evaluation — ROC, precision-recall, confusion matrix, SHAP |
| **Stage 8** | `stage8_validate.py` | Model Validation — cross-validation, threshold optimisation, drift config |
| **Stage 9** | `stage9_ensemble.py` | Ensembling — stacking, voting, and blending comparison |
| **Stage 10** | `stage10_uat.py` | User Acceptance Testing — prediction checks, schema validation, edge cases |
| **Stage 11** | `stage11_deploy.py` | Deployment Packaging — serialised model, scaler, metadata bundle |

> Stages 1, 2, and 5 are orchestration/routing stages handled by the agent system, not standalone scripts.

---

## Supported Problem Types

Brahma automatically detects your problem type from the goal you describe:

| Type | Sub-type | Trigger Keywords |
|------|----------|-----------------|
| **Supervised** | Classification | predict, churn, fraud, detect, classify, risk, score |
| **Supervised** | Regression | forecast, estimate, revenue, price, how much, demand |
| **Unsupervised** | Clustering / Anomaly | segment, cluster, group, find patterns, outlier |
| **Semi-Supervised** | — | partial labels, some labels, few labels |

If ambiguous, Brahma asks a single clarifying question rather than guessing.

---

## Supported Data Sources

Brahma writes the correct connection code automatically for any of the following:

| Category | Formats / Connectors |
|----------|----------------------|
| **Files** | CSV, Excel, Parquet, JSON, XML, TSV |
| **Databases** | PostgreSQL, MySQL, SQLite, MS SQL Server |
| **Cloud** | BigQuery, Snowflake, AWS S3, Azure Blob, Google Cloud Storage |
| **APIs** | REST (HTTP/HTTPS), GraphQL |
| **Spreadsheets** | Google Sheets |

---

## Project Structure

```
ml-super-agent/
│
├── CLAUDE.md                    # Brahma identity & activation rules
│
├── agents/                      # Specialised routing agents
│   ├── super_agent.md           # Master orchestrator (the brain)
│   ├── supervised_learning_agent.md
│   ├── unsupervised_learning_agent.md
│   └── semi_supervised_agent.md
│
├── skills/                      # Reusable skill modules
│   ├── data_ingestion.md
│   ├── data_preprocessing.md
│   ├── eda_analyzer.md
│   ├── feature_engineering.md
│   ├── algorithm_selector.md
│   ├── model_trainer.md
│   ├── model_evaluator.md
│   ├── model_validator.md
│   ├── ensembling.md
│   ├── deployment_tester.md
│   ├── uat_checklist.md
│   └── visualization_style.md
│
├── stage3_eda.py                # EDA script
├── stage4_features.py           # Feature engineering script
├── stage6_train.py              # Training + tuning script
├── stage7_evaluate.py           # Evaluation script
├── stage8_validate.py           # Validation script
├── stage9_ensemble.py           # Ensembling script
├── stage10_uat.py               # UAT script
├── stage11_deploy.py            # Deployment packaging script
│
├── dashboard.py                 # Live pipeline progress dashboard
├── dashboard_log.txt            # Dashboard event log
│
├── data/                        # Input data (not committed in production)
│   └── credit_card_customers.csv
│
└── outputs/                     # All generated artefacts
    ├── charts/
    │   ├── eda/                 # EDA visualisations
    │   ├── training/            # Learning curves, tuning history
    │   ├── evaluation/          # ROC, PR curve, SHAP, confusion matrix
    │   ├── validation/          # CV results, threshold analysis
    │   └── ensembling/          # Ensemble comparison charts
    ├── models/                  # Serialised model files (.pkl)
    └── data/                    # Intermediate pipeline artefacts (.parquet, .pkl)
```

---

## Agents

### Super Agent (`agents/super_agent.md`)
The single entry point. Handles activation, goal collection, problem classification, data source validation, agent routing, and the final pipeline summary banner. The user only ever interacts with the Super Agent directly.

### Supervised Learning Agent (`agents/supervised_learning_agent.md`)
Runs the full supervised pipeline — preprocessing through deployment — for both classification and regression problems.

### Unsupervised Learning Agent (`agents/unsupervised_learning_agent.md`)
Handles clustering (K-Means, DBSCAN, hierarchical) and anomaly detection (Isolation Forest, LOF).

### Semi-Supervised Agent (`agents/semi_supervised_agent.md`)
Handles datasets with partial labels using label propagation and self-training techniques.

---

## Skills

Skills are reusable capability modules that agents call during the pipeline:

| Skill | Purpose |
|-------|---------|
| `data_ingestion` | Reads data from any source type with the correct connector |
| `data_preprocessing` | Handles nulls, outliers, type casting, train/test split |
| `eda_analyzer` | Generates distribution plots, correlation heatmaps, target analysis |
| `feature_engineering` | Encoding, scaling, interaction features, feature selection |
| `algorithm_selector` | Picks the right algorithm family for the problem type and data size |
| `model_trainer` | Trains baseline + tuned models using Optuna |
| `model_evaluator` | Computes metrics, generates SHAP explanations and all evaluation charts |
| `model_validator` | Cross-validation, threshold tuning, data drift configuration |
| `ensembling` | Stacking, voting, and blending; selects the best ensemble |
| `deployment_tester` | Validates the deployment package before sign-off |
| `uat_checklist` | Runs UAT assertions: schema checks, prediction sanity, edge cases |
| `visualization_style` | Enforces consistent chart styling across all pipeline output |

---

## Outputs

After a successful pipeline run, all outputs are written to the `outputs/` directory:

| Output | Location | Description |
|--------|----------|-------------|
| EDA charts | `outputs/charts/eda/` | Target distribution, feature correlations, violin plots |
| Training charts | `outputs/charts/training/` | Learning curves, Optuna optimisation history |
| Evaluation charts | `outputs/charts/evaluation/` | ROC curve, PR curve, confusion matrix, SHAP beeswarm, calibration |
| Validation charts | `outputs/charts/validation/` | CV results, threshold analysis |
| Ensemble charts | `outputs/charts/ensembling/` | Model comparison bar chart |
| Trained models | `outputs/models/` | `.pkl` files for baseline, tuned, and final models |
| Deployment package | `outputs/models/deployment_package.pkl` | Production-ready bundle (model + scaler + metadata) |
| Intermediate data | `outputs/data/` | Preprocessed and feature-engineered `.parquet` files |
| Leaderboard | `outputs/data/leaderboard.csv` | All model scores ranked by primary metric |

---

## Example Session

```
User   : Wake Up Brahma

Brahma : Brahma is awake.
         I am ready to transform your data into intelligence.
         Tell me your goal and your data source.
         Nothing else is required.

User   : Goal: Predict which credit card customers will churn next month.
         Data: data/credit_card_customers.csv

Brahma : Here is what I understand:

         Goal   : Binary classification — predict customer churn (yes/no)
         Data   : FILE — data/credit_card_customers.csv (5.2 MB)
         Output : Trained model, evaluation charts, SHAP explanations, deployment package

         Confirm: Is this correct? (yes / no)

User   : yes

Brahma : ─────────────────────────────────────────
         Problem Type : SUPERVISED
         Sub-type     : Classification
         Routing to   : supervised_learning_agent
         ─────────────────────────────────────────
         Starting pipeline. I will announce each stage as I enter it.

         [Stages 3–11 run automatically]

         ╔══════════════════════════════════════════════════════════╗
         ║              BRAHMA PIPELINE COMPLETE                    ║
         ╠══════════════════════════════════════════════════════════╣
         ║  Goal        : Predict customer churn                    ║
         ║  Stages run  : 9/9                                       ║
         ║  Final model : XGBoost (tuned, stacked ensemble)        ║
         ║  Test score  : ROC-AUC = 0.974                           ║
         ╚══════════════════════════════════════════════════════════╝
```

---

## Error Handling

Brahma follows four non-negotiable error principles:

1. **Never crash silently** — every error prints a plain English explanation.
2. **Never lose completed work** — all outputs are saved before stopping on a blocker.
3. **Always offer recovery** — on any failure: `Retry / Skip / Stop?`
4. **Never guess** — if the goal or data source is ambiguous, Brahma asks before proceeding.

---

## Requirements

The pipeline scripts use the following Python libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
shap
pyarrow      # for .parquet I/O
sqlalchemy   # for database connections
requests     # for API sources
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna shap pyarrow sqlalchemy requests
```

---

## License

This project is proprietary. All rights reserved.
