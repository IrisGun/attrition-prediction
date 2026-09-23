# Attrition Predictor
_Disclaimer: This repo is for demonstration purpose only. No production data provided, no enterprise's status quo addressed_

Attrition Predictor is a full-stack application for early employee attrition risk prediction.
It combines a React dashboard, an Express API server, and a Python ML pipeline to ingest (herein generate), train, evaluate, and serve attrition risk scores.  


## Overview

The system helps HR and operations teams:

- Predict the probability that an active employee may resign within the next 90 days
- Identify high-risk employees early for proactive intervention
- Monitor risk distribution and top-risk cases through a dashboard

## Architecture

The project has three main layers:

- Frontend dashboard with React + Vite
- Backend API with Express + TypeScript
- ML pipeline with Python + scikit-learn

High-level flow:

1. Frontend requests prediction data from backend.
2. Backend serves prediction artifacts from the ML output folder.
3. Users can trigger the full ML pipeline from the dashboard.
4. Pipeline runs data generation, feature engineering, drift detection, training (if needed), and inference.
5. Inference writes JSON outputs consumed by the frontend.

## Repository Structure

- `src/`
  - [src/App.tsx](src/App.tsx): Main dashboard UI
  - [src/main.tsx](src/main.tsx): React entry point
- [server.ts](server.ts): Express server and API routes
- `ml`
  - [ml/run_pipeline.py](ml/run_pipeline.py): End-to-end ML orchestration
  - [ml/data_gen.py](ml/data_gen.py): Synthetic raw data generation
  - [ml/feature_engineering.py](ml/feature_engineering.py): Snapshot-based feature creation
  - [ml/drift_detection.py](ml/drift_detection.py): Drift detection with KS test
  - [ml/models.py](ml/models.py): Model classes and model comparison
  - [ml/train_eval.py](ml/train_eval.py): Training and evaluation workflow
  - [ml/inference.py](ml/inference.py): Risk scoring and prediction artifact export
- [attrition_ml_flow.py](attrition_ml_flow.py): Standalone prototype flow (separate from runtime API flow)
- [main.py](main.py): Minimal Python entry sample
- [vite.config.ts](vite.config.ts): Vite config
- [tsconfig.json](tsconfig.json): TypeScript config
- [package.json](package.json): Node scripts and dependencies
- [metadata.json](metadata.json): App metadata
- [.env.example](.env.example): Environment variable template

## Tech Stack

Frontend:

- React 19
- Vite
- Tailwind CSS
- Recharts
- Framer Motion

Backend:

- Express
- TypeScript
- tsx runtime

Machine Learning:

- pandas
- numpy
- scikit-learn
- scipy
- pickle for model persistence

## Prerequisites

- Node.js (modern LTS recommended)
- Python 3.14+
- uv

## Setup

### 1. Install Node dependencies

```bash
npm install
```

### 2. Configure environment variables

Create an environment file based on [.env.example](.env.example) and provide values needed by your environment.

### 3. Sync Python environment with uv

```bash
uv sync
```

If you need to add ML dependencies into [pyproject.toml](pyproject.toml), use:

```bash
uv add pandas numpy scikit-learn scipy
uv sync
```

## Run Locally

Start the application in development mode:

```bash
npm run dev
```

Server runs on: `http://localhost:3000`

In development, Express runs Vite in middleware mode, so frontend and API are served together.

## Available Scripts

Defined in [package.json](package.json):

- `npm run dev`: Start Express + Vite middleware
- `npm run build`: Build frontend production bundle
- `npm run preview`: Preview built frontend
- `npm run clean`: Remove dist folder
- `npm run lint`: TypeScript type check (no emit)

## API Endpoints

`GET /api/health`

- Returns service status.
- Response example:
  - status: ok

`GET /api/ml/predictions`

- Returns current prediction artifact from ML output.
- Behavior:
  - If prediction artifact exists, returns real data
  - If artifact does not exist yet, returns fallback mock data

`POST /api/ml/run`

- Triggers full ML pipeline execution through a Python subprocess.
- Behavior:
  - Runs the pipeline orchestrator
  - Returns success message and pipeline stdout
  - Returns error details if pipeline fails

## ML Pipeline

Main entry:

- [ml/run_pipeline.py](ml/run_pipeline.py)

Pipeline stages:

1. Data generation
2. Feature engineering
3. Drift detection
4. Training and evaluation (conditional)
5. Inference

For detailed ML documentation, see:

- [ml/README.md](ml/README.md)

## Dashboard Behavior

The dashboard displays:

- Total active employees
- High-risk employee count
- Average risk score
- Risk distribution (pie chart)
- Top high-risk employees (table)

The pipeline run button triggers backend endpoint POST /api/ml/run, then refreshes prediction data.

## Notes

- [attrition_ml_flow.py](attrition_ml_flow.py) is a standalone prototype script and not the primary runtime pipeline used by backend APIs.
- Some dashboard charts currently use static demo values while key prediction cards and the high-risk table are powered by API data.
- Category encoding is simple and may require stronger consistency strategies for production workflows.

## Troubleshooting

Prediction endpoint returns fallback data:

- Run pipeline once using the dashboard button or the direct Python command.

Python module import errors:

- Run `uv sync` to install project dependencies.

Model not found during inference:

- Ensure the training stage completed and the model file exists in ml/models.

Port conflict on 3000:

- Stop the conflicting process or reconfigure the runtime port in server setup.

## Future Improvements

- Declare and pin all Python ML dependencies in [pyproject.toml](pyproject.toml)
- Replace random split with strict time-based split for evaluation realism
- Add robust categorical encoding persistence across train and inference
- Add structured logging and monitoring for each pipeline stage
- Expand dashboard with trend data sourced from real artifacts
