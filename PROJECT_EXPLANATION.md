# Bus Journey Time and Delay Prediction System – Project Explanation
## Tamil Nadu State Transport Corridors (Simulated Student Project)
**Internship Organization:** Career Solutions  
**Student Name / Role:** AIML Student Intern  
**Important Disclaimer:** **SIMULATED DATA – NOT A LIVE TRANSPORT FEED**

---

## 1. Project Title & Overview
**Bus Journey Time and Delay Prediction System – Tamil Nadu Transport**

This project is a beginner-friendly Python machine learning application built for an internship at Career Solutions. It predicts:
1. **Journey Duration:** Total travel time in minutes.
2. **Estimated Arrival Time (ETA):** When the bus will reach its destination based on departure time.
3. **Probability of Being Late:** Probability score from 0.0 to 1.0 (0% to 100%).
4. **Delay Classification:** Whether the bus is **"Late"** (arriving more than 10 minutes past timetable) or **"On Time"**.
5. **Model Comparison:** Comparison of three regression algorithms against the scheduled timetable baseline, and Logistic Regression against a majority-class baseline.

---

## 2. Problem Statement
In public bus transportation, buses often get delayed because of:
- Morning and evening peak-hour traffic jams.
- Sudden heavy monsoon showers.
- Extra passenger boardings at intermediate stops.

Usually, passengers and dispatchers only have static printed timetables, which do not account for traffic or rain. This project uses machine learning to predict actual travel time and late probability the moment the bus leaves the depot.

---

## 3. Technology Stack and Why We Used Each

We deliberately used standard, beginner-friendly Python libraries:
- **Python 3.12:** Clean, readable, and widely used for machine learning.
- **Pandas:** Used to load, clean, inspect, and slice tabular data (`.csv` files).
- **NumPy:** Used for generating random numbers with a fixed seed (`np.random.seed(42)`).
- **Scikit-learn:** Used for preprocessing (`StandardScaler`, `OneHotEncoder`) and training models (`LinearRegression`, `LogisticRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`).
- **Matplotlib:** Generates simple charts (pie chart, bar plots, scatter plot).
- **Streamlit:** Creates a clean web dashboard with straightforward Python code (`st.title()`, `st.button()`, `st.metric()`).
- **SQLite3:** Built-in Python database that saves departure events and predictions to a local file (`bus_predictions.db`).
- **Joblib:** Saves trained models to `.pkl` files so the app can make instant predictions without retraining.

---

## 4. Project Folder Structure

```text
bus-journey-prediction-system/
│
├── data/
│   ├── trips.csv                 <- 3,000 historical trips across 60 days
│   └── departure_events.csv      <- Departure replay feed (no actual duration or late flag)
│
├── models/
│   ├── linear_regression.pkl     <- Duration regressor
│   ├── logistic_regression.pkl   <- Late/On-time classifier
│   ├── decision_tree.pkl         <- Decision Tree Regressor
│   ├── random_forest.pkl         <- Random Forest Regressor
│   ├── best_regression_model.pkl <- Winning model retrained on 80% train+val data
│   ├── preprocessor.pkl          <- Saved scaler & one-hot encoder
│   └── selected_model_info.json  <- Info about the winning model
│
├── reports/
│   ├── validation_metrics.csv    <- Comparison table on validation set
│   └── test_metrics.csv          <- Final evaluation table on test set
│
├── src/
│   ├── generate_data.py          <- Generates the synthetic dataset
│   ├── train.py                  <- Trains and evaluates the 4 models
│   ├── predict.py                <- Prediction pipeline with input checks
│   └── database.py               <- SQLite database functions
│
├── app.py                        <- Streamlit web dashboard
├── test_system.py                <- Test script that checks all 15 requirements
├── requirements.txt              <- Python package list
├── README.md                     <- Quick start guide
├── PROJECT_EXPLANATION.md        <- Complete project explanation
├── SIMPLE_CODE_EXPLANATION.md   <- File-by-file code simplification guide
└── bus_predictions.db            <- Local SQLite database
```

---

## 5. How the Synthetic Dataset Was Generated (`src/generate_data.py`)

Real government bus GPS data is restricted, so we generated realistic synthetic data using Python:
- **Seed:** `42` (`np.random.seed(42)`), ensuring identical numbers every time it runs.
- **Duration:** 60 simulated days (from `2026-01-01` to `2026-03-01`).
- **Daily Volume:** 50 trips per day = **3,000 total trips**.
- **Five Fictional Tamil Nadu Routes:**
  1. `TN-DEMO-01`: Chennai (CMBT) to Vellore (138 km, 10 stops, 180 scheduled min)
  2. `TN-DEMO-02`: Coimbatore to Salem (164 km, 14 stops, 210 scheduled min)
  3. `TN-DEMO-03`: Madurai to Trichy (132 km, 9 stops, 170 scheduled min)
  4. `TN-DEMO-04`: Tirunelveli to Madurai (158 km, 12 stops, 200 scheduled min)
  5. `TN-DEMO-05`: Chennai (CMBT) to Villupuram (152 km, 11 stops, 195 scheduled min)

### Two Datasets Generated:
1. `data/trips.csv`: Historical dataset containing inputs plus ground truth (`actual_duration_min` and `late_flag`). Used for model training.
2. `data/departure_events.csv`: Simulated departure feed containing only variables known at departure time. Used for replay testing. **Zero data leakage!**

---

## 6. Explanation of Every Dataset Column

| Column Name | Data Type | Meaning |
| :--- | :--- | :--- |
| `trip_id` | String | Unique trip identifier (e.g., `TRIP_0001`). |
| `route_id` | Text | Bus corridor route code (`TN-DEMO-01` to `TN-DEMO-05`). |
| `departure_date` | Date | Date of departure (`YYYY-MM-DD`). |
| `route_distance_km` | Float | Route distance in kilometers (approx. 130 to 165 km). |
| `scheduled_duration_min` | Integer | Standard timetable travel time in minutes. |
| `departure_hour` | Integer | Hour of departure in 24-hr time (5 AM to 10 PM). |
| `weekday` | Text | Day of the week (Monday to Sunday). |
| `stop_count` | Integer | Number of intermediate stops along the route. |
| `departure_traffic_index` | Float | Congestion score from 0.0 (clear) to 1.0 (gridlock). |
| `rain_mm_last_hour` | Float | Rainfall in the last hour in millimeters (0 to 35 mm). |
| `actual_duration_min` | Float | True trip duration in minutes. **Target for regression; NEVER used as input.** |
| `late_flag` | Binary (0/1) | 1 if actual duration > scheduled + 10 min, else 0. **Target for classification; NEVER used as input.** |

---

## 7. How Actual Duration and Late Flag Are Calculated

### Formula for Actual Journey Duration:
$$\text{actual\_duration\_min} = \text{scheduled\_duration\_min} + \text{traffic\_delay} + \text{rain\_delay} + \text{stop\_delay} + \text{noise}$$

In code:
```python
traffic_delay = (departure_traffic_index - 0.28) * 32.0
rain_delay = (rain_mm_last_hour / 35.0) * 16.0
stop_delay = (stop_count - route["base_stops"]) * 1.8
random_noise = np.random.normal(loc=0.0, scale=3.5)

actual_duration_min = scheduled_duration_min + traffic_delay + rain_delay + stop_delay + random_noise
```
- **Traffic Delay:** Average traffic is around 0.28. Higher traffic adds up to 22 extra minutes.
- **Rain Delay:** Heavy monsoon rainfall (up to 35 mm) adds up to 16 minutes.
- **Stop Delay:** Each unexpected additional stop adds ~1.8 minutes.
- **Noise:** Random vehicle/driver variation using a normal distribution (+/- 3.5 min).

### Formula for Late Flag:
```python
if actual_duration_min > (scheduled_duration_min + 10.0):
    late_flag = 1
else:
    late_flag = 0
```
- **On Time (`late_flag = 0`):** 2,108 trips (70.3%)
- **Late (`late_flag = 1`):** 892 trips (29.7%)

---

## 8. Data Inspection and Preprocessing (`src/train.py`)

### 1. Inspection:
- Checked shape: 3,000 rows, 12 columns.
- Checked null values: `df.isnull().sum()` found **0 null values**.
- Checked duplicate rows: `df.duplicated().sum()` found **0 duplicates**.
- Outlier detection with IQR: Found **0 corrupt outliers**; extreme rush-hour delays are genuine transport behavior, so they are kept.

### 2. Chronological Split (60% / 20% / 20%):
Transport data is time-dependent. We **do not shuffle** randomly because future information would leak into past predictions.
- **Train Set (First 60% of dates):** 36 days (1,800 trips)
- **Validation Set (Next 20% of dates):** 12 days (600 trips) – used to select the best regression model.
- **Test Set (Final 20% of dates):** 12 days (600 trips) – kept completely untouched until final evaluation.

### 3. Encoding and Scaling:
Using Scikit-learn's `ColumnTransformer`:
- **Categorical Columns (`route_id`, `weekday`):** Transformed into binary columns using `OneHotEncoder(handle_unknown='ignore')`.
- **Numerical Columns:** Standardized using `StandardScaler()` so all features have mean 0 and standard deviation 1.
- **No Data Leakage Rule:** The preprocessor is fitted **only on the training set**. Validation and test data are only transformed.

---

## 9. The Four Machine Learning Algorithms

### 1. Linear Regression
- **Purpose:** Predicts journey duration in minutes.
- **How it works:** Finds the best-fit line by calculating weights for each feature:
  $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots$$
- **Code:** `model_lr = LinearRegression(); model_lr.fit(X_train, y_train)`

### 2. Logistic Regression
- **Purpose:** Predicts probability of arriving late and assigns `"Late"` or `"On Time"`.
- **How it works:** Passes a linear combination through the Sigmoid function:
  $$P(\text{Late}) = \frac{1}{1 + e^{-z}}$$
- **Code:** `model_log = LogisticRegression(random_state=42); model_log.fit(X_train, y_train_late)`
- Uses `model_log.predict_proba()` to get late probabilities. Default threshold is `0.50`.

### 3. Decision Tree Regressor
- **Purpose:** Predicts journey duration by splitting data based on if-else rules.
- **Parameters:** `DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)`
- Good for capturing non-linear thresholds (e.g. traffic > 0.7 causes a sudden jump in delay).

### 4. Random Forest Regressor
- **Purpose:** Predicts journey duration using an ensemble of 100 decision trees.
- **Parameters:** `RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)`
- Averages predictions across 100 trees to reduce error and variance.

---

## 10. Real Evaluation Results

### Validation Set Comparison (600 Trips):
*Used to choose the best regression model based on lowest Validation MAE:*

| Model | Type | Validation MAE (min) | Validation RMSE (min) |
| :--- | :--- | :--- | :--- |
| **Timetable Scheduled Baseline** | Baseline | 7.889 min | 9.845 min |
| **Linear Regression (Winner)** | Regression | **2.770 min** | **3.451 min** |
| **Random Forest Regressor** | Regression | 3.087 min | 3.891 min |
| **Decision Tree Regressor** | Regression | 4.044 min | 5.067 min |

**Decision:** **Linear Regression** achieved the lowest error (**2.770 minutes**). It was retrained on combined Train + Validation data (80% / 2,400 trips).

### Logistic Regression Validation Performance:
- **Accuracy:** **90.33%** (vs Majority-Class Baseline: 67.83%)
- **Precision:** **0.8689**
- **Recall:** **0.8238**
- **F1-Score:** **0.8457**

### Final Evaluation on Untouched Test Set (600 Trips):
*Evaluated only once after retraining the winning model on 80% data:*

| Metric | Model Result | Baseline Result | Improvement |
| :--- | :--- | :--- | :--- |
| **Regression MAE** | **2.891 min** | 8.072 min | **64.2% error reduction** |
| **Regression RMSE** | **3.578 min** | 10.113 min | **64.6% error reduction** |
| **Classification Accuracy** | **86.33%** | 66.83% | **+19.50% gain** |
| **Classification F1-Score** | **0.7897** | 0.0000 | **+0.7897 gain** |
| **Classification Precision** | **0.8063** | 0.0000 | **+0.8063 gain** |
| **Classification Recall** | **0.7739** | 0.0000 | **+0.7739 gain** |

---

## 11. Prediction Pipeline & Safety Checks (`src/predict.py`)

The function `predict_bus_journey()` handles prediction safely:
1. **Input Validation:**
   - Distance $\le 0$ $\rightarrow$ Error
   - Scheduled duration $\le 0$ $\rightarrow$ Error
   - Rain $< 0$ $\rightarrow$ Error
   - Traffic index outside $[0.0, 1.0]$ $\rightarrow$ Error
   - Unknown route $\rightarrow$ Warning for manual dispatcher review
2. **Stale Feed Fallback:**
   - If an automated feed is older than 120 seconds, it falls back to the scheduled timetable arrival time and marks `"Scheduled (Fallback)"`.
3. **Feature Derivation:**
   - Extracts `departure_hour` and `weekday` from the departure timestamp.
4. **Prediction Calculation:**
   - Predicts duration with Linear Regression.
   - Calculates arrival time: $\text{Departure Time} + \text{Predicted Duration}$.
   - Predicts probability of delay with Logistic Regression.
   - Assigns `"Late"` if probability $\ge 0.50$, else `"On Time"`.

---

## 12. SQLite Database Logging (`src/database.py`)

Uses two simple tables in `bus_predictions.db`:
1. `trip_events`: Stores raw trip inputs (`trip_id`, `route_id`, `departure_timestamp`, `input_data`, `source`).
2. `predictions`: Stores prediction outputs (`trip_id`, `predicted_duration_min`, `predicted_arrival`, `late_probability`, `late_label`, `model_version`, `predicted_at`).

Every manual prediction and replayed departure is automatically logged and viewable on Page 6 of the dashboard.

---

## 13. Streamlit Dashboard Features (`app.py`)

Built using basic Streamlit components:
- **Page 1: Dataset & Data Dictionary:** High-level metrics, CSV download buttons, local file paths, dataset preview, and column dictionary.
- **Page 2: Bus Departure Prediction:** Manual input form to predict journey time and late probability with instant feedback.
- **Page 3: Simulated Departure Replay:** Step through simulated departure events one by one, with an optional 30-second auto-refresh.
- **Page 4: Model Comparison & Metrics:** Displays validation and test benchmark tables.
- **Page 5: Transport Visualizations:** Matplotlib pie chart, duration bar chart, MAE bar chart, and scatter plot.
- **Page 6: Prediction History:** Live table of all past predictions stored in SQLite.

---

## 14. How to Run the Project

Open PowerShell in the project directory:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate simulated data (Seed: 42)
python src/generate_data.py

# 3. Train all 4 models and save reports
python src/train.py

# 4. Run automated verification tests
python test_system.py

# 5. Launch the Streamlit dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 15. Limitations and Future Scope
- **Simulated Data:** The routes and weather are synthetic; this is an educational prototype and does not connect to live government GPS.
- **Terminal Departure Prediction:** Forecasts are produced when the bus departs; it does not update en-route if an unexpected jam occurs midway.
- **Future Real-World Connection:** In a production system, this could connect to **GTFS-Realtime** vehicle GPS feeds and open weather APIs (IMD/OpenWeather).
