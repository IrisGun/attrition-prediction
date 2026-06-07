# ML Pipeline Documentation

This document contains detailed technical documentation for the Machine Learning workflow used in Attrition Predictor.

## Scope

This ML subsystem is responsible for:

- Generating synthetic HR and behavioral datasets for experimentation (TODO: Implement Data Ingestion instead of generating synthetic data)
- Building snapshot-based training features
- Detecting data drift between reference and current snapshots
- Training and selecting a classification model for attrition risk
- Producing dashboard-ready prediction artifacts

Main orchestration entrypoint:

- [run_pipeline.py](run_pipeline.py)

## End-to-End Pipeline

The orchestrator executes these stages in order:

1. Data ingestion (herein generation for demonstration purpose)
2. Feature engineering
3. Drift detection
4. Training and evaluation (conditional)
5. Inference and serving artifact generation

## Pipeline Orchestration Logic

File:

- [run_pipeline.py](run_pipeline.py)

Behavior:

- Always regenerates raw and processed data for configured snapshots
- Runs drift detection before deciding retraining
- Retrains when either condition is true:
  - Drift is detected
  - No trained model exists at ml/models/best_model.pkl
- Always runs inference at the end and refreshes ml/artifacts/predictions.json

## Stage 1: Data Generation

File:

- [data_gen.py](data_gen.py)

Purpose:

- Build synthetic raw datasets that simulate employee lifecycle and financial behavior
- Note: Solely for demonstration purpose. TODOs:
  - Add data pipeline to ingest data from DataMart
  - Adjust data preprocessing accordingly

Generated raw tables:

- employee_master.csv: employee profile and demographics
- attendance.csv: daily attendance and overtime history
- ewa_usage.csv: Earned Wage Access transaction history
- resignation.csv: resignation events used for supervised labeling

Simulation assumptions:

- Baseline attrition is around 15 percent
- Working days are Monday to Saturday
- App usage is heterogeneous, including non-users and frequent users

Output:

- ml/data/raw

## Stage 2: Feature Engineering

File:

- [feature_engineering.py](feature_engineering.py)

Purpose:

- Convert raw event data into model-ready snapshot features

Current snapshot schedule in script entrypoint:

- 2025-01-01
- 2025-06-01
- 2025-12-01
- 2026-03-01 (current inference snapshot)

Active employee definition at snapshot date S:

- join_date <= S
- employee is not resigned on or before S

Feature groups:

- Tenure:
  - tenure_days
  - tenure_months
- Attendance in last 30 days:
  - days_worked_last_30
- EWA in last 60 days:
  - ewa_count_60
  - ewa_sum_60
  - ewa_avg_60
- Encoded categoricals:
  - company_cat
  - dept_cat
  - city_cat

Label construction:

- resigned_next_3m = 1 if resignation date is in (S, S + 90 days]
- resigned_next_3m = 0 otherwise

Output:

- ml/data/processed/features_YYYY-MM-DD.csv

## Stage 3: Drift Detection

File:

- [drift_detection.py](drift_detection.py)

Purpose:

- Identify statistical distribution shift before retraining decisions

Default snapshot comparison:

- Training reference snapshot: 2025-12-01
- Current snapshot: 2026-03-01

Method:

- Two-sample Kolmogorov-Smirnov test for each monitored feature
- A feature is flagged drifted when p-value < 0.05

Monitored features:

- tenure_days
- days_worked_last_30
- ewa_count_60
- ewa_sum_60
- ewa_avg_60

Output:

- ml/artifacts/drift_results.json

## Stage 4: Training and Evaluation

Files:

- [train_eval.py](train_eval.py)
- [models.py](models.py)

Purpose:

- Train multiple candidate models, compare metrics, persist best model

Training snapshots:

- 2025-01-01
- 2025-06-01
- 2025-12-01

Candidate models:

- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression

Metrics:

- Precision
- Recall
- F1
- ROC-AUC

Selection:

- Best model is selected by highest F1 score on held-out split

Persisted outputs:

- Best model: ml/models/best_model.pkl
- Full comparison: ml/artifacts/model_comparison.json
- Best metrics summary: ml/artifacts/metrics.json

## Stage 5: Inference

File:

- [inference.py](inference.py)

Purpose:

- Score active employees and produce frontend-consumable predictions

Input snapshot:

- Default: 2026-03-01 processed feature file

Scoring logic:

- Uses predict_proba when available
- Converts probability to integer percentage risk_score from 0 to 100

Risk buckets:

- High Risk: risk_score >= 70
- Medium Risk: 40 <= risk_score < 70
- Low Risk: risk_score < 40

Top-risk extraction:

- Select top 5 employees by descending risk_score

Output:

- ml/artifacts/predictions.json

## Data and Artifact Contract

Raw datasets:

- ml/data/raw/employee_master.csv
- ml/data/raw/attendance.csv
- ml/data/raw/ewa_usage.csv
- ml/data/raw/resignation.csv

Processed snapshots:

- ml/data/processed/features_YYYY-MM-DD.csv

Model artifact:

- ml/models/best_model.pkl

Monitoring and evaluation artifacts:

- ml/artifacts/metrics.json
- ml/artifacts/model_comparison.json
- ml/artifacts/drift_results.json

Serving artifact:

- ml/artifacts/predictions.json

## Prediction JSON Schema

The backend endpoint GET /api/ml/predictions returns content shaped from ml/artifacts/predictions.json with keys:

- total_employees: integer
- high_risk_count: integer
- avg_risk_score: integer
- risk_distribution: list of objects with name, value, color
- top_high_risk: list of employee objects

Each top_high_risk object currently includes:

- emp_id
- company
- dept
- tenure_months
- risk_score

## Running ML Stages with uv

Run stages independently:

```bash
uv run python ml/data_gen.py
uv run python ml/feature_engineering.py
uv run python ml/drift_detection.py
uv run python ml/train_eval.py
uv run python ml/inference.py
```

Run full pipeline:

```bash
uv run python ml/run_pipeline.py
```

## Operational Notes

- The pipeline currently uses synthetic data for demonstration and integration testing.
- The train/test split is random, not strictly temporal.
- Categorical encoding uses snapshot-local category codes and may be inconsistent across time windows.
- Retraining trigger is based on selected numeric features only.

## Suggested Next Hardening Steps

- Persist encoder mappings for stable categorical handling across train and inference
- Move from random split to temporal split for realistic offline evaluation
- Add threshold calibration and business-specific decision policies
- Add model versioning and rollback policy for production operations
- Add structured logging per stage and data quality checks before training
