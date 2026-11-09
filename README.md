# 🚌 Bus Journey Time and Delay Prediction System – Tamil Nadu Transport

> **⚠️ SIMULATED DATA – NOT A LIVE TRANSPORT FEED**  
> *This student internship project uses synthetic data generated for fictional Tamil Nadu bus corridors. It does not connect to actual government transport APIs or live vehicle GPS trackers.*

---

## 📌 Project Overview

This project is a beginner-friendly Python machine learning application developed for an AIML internship at **Career Solutions / SkillFirstLabs**. It predicts:
1. **Journey Duration:** Total travel time in minutes.
2. **Estimated Arrival Time (ETA):** When the bus will reach its destination based on departure time.
3. **Probability of Being Late:** Probability score from 0.0 to 1.0 (0% to 100%).
4. **Delay Classification:** Whether the bus is **"Late"** (arriving more than 10 minutes past timetable) or **"On Time"**.
5. **Model Comparison:** Comparison of three regression algorithms against the scheduled timetable baseline, and Logistic Regression against a majority-class baseline.

---

## 📁 Simple Project Folder Structure

```text
bus-journey-prediction-system/
│
├── app.py                     # Streamlit web dashboard (6 clean sections)
├── generate_data.py           # Generates 3,000 synthetic bus trips across 60 days
├── train_models.py            # Trains 4 ML models using chronological splitting
├── predict.py                 # Real-time prediction function with safety checks
├── database.py                # SQLite database management (bus_predictions.db)
├── predict_cli.py             # Command-line demo tool for quick terminal testing
├── test_system.py             # Automated 15-point test suite
├── requirements.txt           # Minimal dependencies
├── README.md                  # Quickstart guide
├── SIMPLE_CODE_EXPLANATION.md # Student code guide & viva Q&A
├── PROJECT_EXPLANATION.md     # In-depth viva documentation
│
├── data/
│   ├── trips.csv              # Historical trips dataset (3,000 rows)
│   └── departure_events.csv   # Replay departure feed (no data leakage)
│
├── models/
│   ├── linear_regression.pkl  # Trained Linear Regression model
│   ├── logistic_regression.pkl# Trained Logistic Regression model
│   ├── decision_tree.pkl      # Decision Tree Regressor
│   ├── random_forest.pkl      # Random Forest Regressor
│   ├── best_regression_model.pkl # Winning model retrained on 80% train+val
│   ├── preprocessor.pkl       # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── selected_model_info.json # Metadata of selected model
│
├── reports/
│   ├── validation_metrics.csv # Validation comparison table
│   └── test_metrics.csv       # Untouched holdout test evaluation table
│
└── bus_predictions.db         # Local SQLite database
```

---

## 🚀 Quick Start Guide

### 1. Install Requirements
```powershell
pip install -r requirements.txt
```

### 2. Generate Simulated Data (Seed: 42)
```powershell
python generate_data.py
```

### 3. Train All Four Machine Learning Models
```powershell
python train_models.py
```

### 4. Run the Automated Verification Tests
```powershell
python test_system.py
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### Optional: Test via Terminal (No Browser Required)
```powershell
python predict_cli.py
```

---

## 📊 Real Experimental Results (Seed: 42)

### 1. Validation Set Model Comparison (600 Trips)
*Used to select the winning regression model based on lowest Validation MAE:*

| Model | Type | Validation MAE (min) | Validation RMSE (min) |
| :--- | :--- | :--- | :--- |
| **Timetable Scheduled Baseline** | Baseline | 7.889 min | 9.845 min |
| **Linear Regression (Winner)** | Regression | **2.770 min** | **3.451 min** |
| **Random Forest Regressor** | Regression | 3.087 min | 3.891 min |
| **Decision Tree Regressor** | Regression | 4.044 min | 5.067 min |

*Linear Regression achieved the lowest validation MAE (2.770 min) and was retrained on combined Train + Validation data (80%).*

### 2. Final Evaluation on Untouched Test Set (20% Holdout / 600 Trips)
| Evaluation Metric | Model Result | Baseline Result | Improvement |
| :--- | :--- | :--- | :--- |
| **Regression MAE** | **2.891 min** | 8.072 min | **64.2% error reduction** |
| **Regression RMSE** | **3.578 min** | 10.113 min | **64.6% error reduction** |
| **Classification Accuracy** | **86.33%** | 66.83% | **+19.50% gain** |
| **Classification F1-Score** | **0.7897** | 0.0000 | **+0.7897 gain** |
| **Classification Precision** | **0.8063** | 0.0000 | **+0.8063 gain** |
| **Classification Recall** | **0.7739** | 0.0000 | **+0.7739 gain** |

---

## 📖 In-Depth Guides for Mentors & Viva

- **Student-Friendly Code Guide:** [`SIMPLE_CODE_EXPLANATION.md`](file:///c:/Users/gpdsa/Downloads/bus-journey-prediction-system/SIMPLE_CODE_EXPLANATION.md)
- **Detailed 15-Section Viva Report:** [`PROJECT_EXPLANATION.md`](file:///c:/Users/gpdsa/Downloads/bus-journey-prediction-system/PROJECT_EXPLANATION.md)
- **Downloadable Standalone ZIP:** [`bus-journey-prediction-system.zip`](file:///c:/Users/gpdsa/Downloads/bus-journey-prediction-system/bus-journey-prediction-system.zip)
