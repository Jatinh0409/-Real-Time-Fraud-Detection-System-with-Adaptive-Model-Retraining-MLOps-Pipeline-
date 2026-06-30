# Real-Time Fraud Detection System with Drift Monitoring

An end-to-end MLOps pipeline that detects fraudulent transactions in real time,
monitors incoming data for distribution drift, and flags when the model needs
retraining — built to mirror how fraud detection systems work in production.

## Why this project

Most student ML projects stop at "train a model in a notebook." This one
covers the parts that actually matter in industry: handling severe class
imbalance correctly, serving predictions with low latency via a REST API,
containerizing for deployment, and monitoring a live model for staleness.

## Architecture

```
data/generate_data.py   → synthetic PaySim-style transaction generator
                           (swap for IEEE-CIS/PaySim real data with zero code changes)
models/train_model.py   → XGBoost pipeline with class-imbalance handling,
                           feature engineering, and PR-AUC/F1 evaluation
app/app.py               → FastAPI service: /predict, /health, /metrics
models/monitor.py        → KS-test + PSI based drift detector (Phase 2)
Dockerfile                → containerized API, deployable anywhere
```

## Results (on synthetic data — swap in real dataset for production-credible numbers)

| Metric | Value |
|---|---|
| ROC-AUC | 1.00* |
| PR-AUC | 0.9998* |
| Precision (fraud class) | 0.98 |
| Recall (fraud class) | 0.997 |
| Inference latency | ~4-7ms |

*Synthetic data is more cleanly separable than real fraud data. Expect
0.85–0.95 ROC-AUC on IEEE-CIS/PaySim — which is the more credible number to
quote on a resume.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (or drop in a real dataset with the same columns)
cd data && python generate_data.py --n_rows 150000 --out transactions.csv

# 2. Train
cd ../models && python train_model.py --data ../data/transactions.csv --out model.pkl

# 3. Serve
cd ../app && uvicorn app:app --reload --port 8000
# then POST to http://localhost:8000/predict

# 4. Check for drift on a new batch
cd ../models && python monitor.py --reference ../data/transactions.csv --current ../data/new_batch.csv
```

## Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## Key engineering decisions

- **Class imbalance**: handled via `scale_pos_weight` in XGBoost rather than
  naive oversampling, which avoids inflating false positives.
- **Metric choice**: PR-AUC and F1 on the fraud class are reported instead of
  accuracy, since accuracy is meaningless on a ~1% positive-rate problem.
- **Feature engineering**: balance-drain ratios and account-zeroing flags,
  which are strong fraud signals in transfer-type transactions.
- **Drift detection**: Kolmogorov-Smirnov test per numeric feature and
  Population Stability Index for the categorical feature — both standard,
  explainable drift metrics used in production ML monitoring.

## Next steps to extend (Phase 3)

- Wire `monitor.py`'s drift flag to automatically trigger `train_model.py`
  via Airflow or a cron job, and only promote the new model if it beats the
  current one on a holdout set.
- Replace the synthetic generator with a Kafka producer for true streaming.
- Add Grafana/Evidently AI dashboards for live monitoring visualization.

