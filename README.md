# AI-Based Network Attack Forecasting (SIH 2026 — PS 26153)

**Team:** Code Storm
**Team Members:** Lakshmi Bhat, Mallikarjun, Supriya, Nisha Kumari, Sandeepa, P Prasad

## Problem Statement
Forecasting network attacks *before* they fully materialize, using sequences of
network traffic behavior rather than classifying isolated flows.

## What This MVP Demonstrates
This is a 40% internal-hackathon MVP focused on proving the core forecasting
concept end-to-end using real data and real results — not the full proposed
architecture (see Future Work below).

**Pipeline implemented:**
CIC-IDS-2018 → Preprocessing → Feature Selection → Normalization →
10-Step Time Windows → LSTM → One-Step-Ahead Attack Prediction →
Attack Probability Timeline → Streamlit Dashboard

## Dataset
- Source: CIC-IDS-2018 (Friday-02-03-2018), officially hosted on AWS Open Data
  (`s3://cse-cic-ids2018`)
- Working subset: 132,681 rows — a chronological slice spanning hours 9–11
  (captures a genuine calm-to-attack transition), sampled every 3rd row for
  faster iteration
- Labels: 95,014 Benign / 37,667 Bot (attack)

## Method
- **Preprocessing:** removed infinite/missing values (0.77% of rows), dropped
  10 constant columns, kept 69 features
- **Split:** chronological train/val/test (70/15/15) — **not random** — to
  avoid temporal data leakage. Network traffic is time-ordered, so a model
  must only ever learn from the past and be tested on the future, never the
  reverse.
- **Normalization:** StandardScaler fit only on training data, then applied
  to validation/test — prevents information from leaking across the split
- **Sequences:** length 10 — the past 10 flows are used to predict whether
  the *next* flow is malicious (one-step-ahead forecasting)
- **Model:** single-layer LSTM (64 units) → Dense(1) → Sigmoid
- **Framework:** TensorFlow/Keras
- **Training:** 5 epochs, batch size 128

## Results (on held-out, chronologically-later test set, 19,893 sequences)
| Metric | Value |
|---|---|
| Accuracy | 67.52% |
| Precision | 52.75% |
| Recall | 48.83% |
| F1 Score | 50.71% |
| False Positive Rate | 22.76% |

These are real, measured results from running the code — not fabricated.
Confusion matrix:

|              | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | 10,108 | 2,978 |
| **Actual Attack** | 3,483 | 3,324 |

## Dashboard
A Streamlit dashboard that loads the trained model, accepts an uploaded
traffic CSV, scales and sequences it, and produces live attack predictions.

Run locally:
```bash
cd dashboard
streamlit run app.py
```
Upload a traffic CSV → the dashboard scales features, builds sequences, runs
the LSTM, and shows attack probability, predictions, a probability timeline,
and a downloadable results CSV.

## Future Work (explicitly out of scope for this MVP)
This MVP intentionally does **not** implement the following — they remain
part of the original proposed solution and are planned as future enhancements:
- Multi-step (K-step) forecasting — currently one-step-ahead only
- Full "world model" architecture
- Temporal Transformer / GNN-based modeling
- MITRE ATT&CK technique mapping
- SHAP-based explainability
- Real-time packet capture (currently offline/batch CSV-based)
- Blockchain integration, cloud/enterprise deployment

## Repository Structure
```
notebooks/    → data exploration, preprocessing, and training (Colab)
models/       → saved trained LSTM (.keras) + fitted scaler (.pkl)
dashboard/    → Streamlit app (app.py)
data/         → sample traffic CSV subset
results/      → evaluation plots and metrics
requirements.txt → exact Python package versions used
```

## Tech Stack
Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Streamlit

## References
- Problem Statement 26153 — AI based Network Attack Forecasting from Network
  Traffic Data
- Canadian Institute for Cybersecurity (CIC) — CIC-IDS-2018 dataset
