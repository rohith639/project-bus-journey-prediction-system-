# Simplified Code Guide for College Presentation & Mentor Review
## Bus Journey Time and Delay Prediction System – Tamil Nadu Transport

This document was created specifically to help an AIML student understand, explain, and walk through every file of this project during internship evaluation and viva voce.

---

## 1. Summary of Files Simplified

We inspected all project files and simplified each one into clear, beginner-friendly Python:

| File Name | Purpose | What Complicated Code Was Removed / Simplified |
| :--- | :--- | :--- |
| `src/generate_data.py` | Generates 3,000 synthetic bus trips across 60 days | Replaced complex nested dictionaries and lookups with straightforward loop variables (`route_distance_km`, `traffic_delay`, `rain_delay`). Clear step-by-step formulas. |
| `src/train.py` | Inspects data, splits chronologically, trains 4 models, and evaluates | Removed complicated pipelines and excessive decorators. Kept standard Scikit-learn commands (`.fit()`, `.predict()`). Simple chronological date splitting with pandas `.isin()`. |
| `src/predict.py` | Receives departure inputs and returns predictions | Removed heavy caching decorators and nested exception handlers. Uses a simple `load_models()` function, basic `if/else` input validation, and returns a clean Python dictionary. |
| `src/database.py` | Logs departure events and prediction history in SQLite | Replaced complex connection pools with basic Python `sqlite3` commands (`execute()`, `commit()`, `fetchall()`). |
| `app.py` | Streamlit web application | Removed all raw HTML/CSS styling hacks, complex nested layouts, and custom decorators. Uses standard Streamlit components (`st.title()`, `st.metric()`, `st.dataframe()`, `st.pyplot()`, `st.download_button()`). |
| `PROJECT_EXPLANATION.md` | Full project documentation | Rewritten in clear student-friendly English without academic jargon, explaining each concept simply. |

---

## 2. File-by-File Code Walkthrough

### A. `src/generate_data.py` (Data Generation)
- **What it does:** Generates `data/trips.csv` (historical training data) and `data/departure_events.csv` (replayed departure feed).
- **Key simple variables:**
  - `start_date = datetime(2026, 1, 1)`: Simulation start date.
  - `total_days = 60`, `trips_per_day = 50`: Exactly 3,000 total trips.
  - `np.random.seed(42)`: Guarantees reproducible numbers every time.
- **Duration Formula (how delay is calculated):**
  ```python
  traffic_delay = (departure_traffic_index - 0.28) * 32.0
  rain_delay = (rain_mm_last_hour / 35.0) * 16.0
  stop_delay = (stop_count - route["base_stops"]) * 1.8
  random_noise = np.random.normal(loc=0.0, scale=3.5)

  actual_duration_min = scheduled_duration_min + traffic_delay + rain_delay + stop_delay + random_noise
  ```
- **Late Flag Rule:**
  ```python
  if actual_duration_min > (scheduled_duration_min + 10.0):
      late_flag = 1   # Late
  else:
      late_flag = 0   # On time
  ```
- **Zero Leakage Rule:** `departure_events.csv` contains only departure-time inputs. It does **not** include `actual_duration_min` or `late_flag`.

---

### B. `src/train.py` (Data Preprocessing and Training)
- **What it does:** Trains the four machine learning models and saves evaluation metrics.
- **Key Steps:**
  1. **Data Inspection:** Uses `df.isnull().sum()` to check missing values, `df.duplicated().sum()` for duplicates, and IQR outlier boundaries.
  2. **Chronological Splitting:**
     ```python
     unique_dates = sorted(df['departure_date'].unique())
     train_end = int(0.60 * total_dates)
     val_end = int(0.80 * total_dates)
     ```
     - 60% Train (1,800 trips)
     - 20% Validation (600 trips)
     - 20% Test (600 trips)
  3. **Preprocessing:**
     ```python
     preprocessor = ColumnTransformer(transformers=[
         ('num', StandardScaler(), numerical_features),
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
     ])
     preprocessor.fit(X_train)  # Fitted ONLY on training set!
     ```
  4. **The Four Models:**
     ```python
     # 1. Linear Regression (Predicts journey duration)
     model_lr = LinearRegression()
     model_lr.fit(X_train_proc, y_train_duration)

     # 2. Logistic Regression (Predicts late probability)
     model_log = LogisticRegression(random_state=42, max_iter=1000)
     model_log.fit(X_train_proc, y_train_late)

     # 3. Decision Tree Regressor
     model_dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
     model_dt.fit(X_train_proc, y_train_duration)

     # 4. Random Forest Regressor
     model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
     model_rf.fit(X_train_proc, y_train_duration)
     ```
  5. **Model Selection:**
     - Computes MAE on validation set. Linear Regression achieved the lowest validation MAE (**2.770 min**).
     - Retrains the winning model on Train + Validation data (80%).
     - Evaluates once on untouched Test set (20%) and saves reports to `reports/validation_metrics.csv` and `reports/test_metrics.csv`.

---

### C. `src/predict.py` (Inference Pipeline)
- **What it does:** Accepts departure inputs and returns predicted duration and delay status.
- **Key function: `predict_bus_journey(...)`:**
  1. Validates inputs (distance > 0, duration > 0, rain $\ge 0$, traffic $\in [0, 1]$).
  2. Automatically derives `departure_hour` and `weekday` from the departure time.
  3. Preprocesses inputs using the fitted `preprocessor.pkl`.
  4. Predicts duration using `reg_model.predict()`.
  5. Predicts late probability using `clf_model.predict_proba()[:, 1]`.
  6. Calculates estimated arrival: `departure_time + timedelta(minutes=duration)`.
  7. Returns a simple Python dictionary.

---

### D. `src/database.py` (SQLite Database)
- **What it does:** Stores trip events and predictions in `bus_predictions.db`.
- **Two Simple Tables:**
  - `trip_events`: logs route, departure time, and environmental inputs.
  - `predictions`: logs predicted duration, arrival time, late probability, and model version.
- **Key functions:**
  - `init_db()`: Creates the tables if they don't already exist.
  - `log_trip_event_and_prediction()`: Saves one trip event and prediction.
  - `get_prediction_history()`: Reads the most recent predictions using `pd.read_sql_query()`.
  - `get_prediction_count()`: Returns the total count of predictions.

---

### E. `app.py` (Streamlit Web Dashboard)
- **What it does:** The dispatcher web dashboard.
- **Navigation (6 Pages):**
  1. **Dataset & Data Dictionary:** Metrics, CSV download buttons, preview, column explanations.
  2. **Bus Departure Prediction:** Form to enter route, traffic, rain, and time; shows instant ETA and late probability.
  3. **Simulated Departure Replay:** Step through replayed departures with Next/Previous buttons and 30-second auto-refresh.
  4. **Model Comparison & Metrics:** Tables comparing all 4 models and holdout test metrics.
  5. **Transport Visualizations:** 4 simple Matplotlib charts.
  6. **Prediction History:** Table of past predictions from SQLite.

---

## 3. Features That Could NOT Be Simplified (And Why)

To protect the integrity of the project and meet all internship rubric criteria, these features were preserved exactly as required:

1. **Chronological 60/20/20 Splitting:** We cannot use simple random shuffling (`train_test_split(shuffle=True)`) because in transport systems, future dates must never leak into training past data.
2. **Zero Data Leakage:** `actual_duration_min` and `late_flag` must never be present in `departure_events.csv` or used as input features.
3. **Four Distinct ML Algorithms:** All four algorithms (Linear Regression, Logistic Regression, Decision Tree, Random Forest) must be trained and compared against baselines.
4. **Input Validation & 120-Second Fallback:** Safety checks for negative numbers and the 120-second stale feed fallback are critical safety requirements for transport dispatch systems.
5. **Visible SIMULATED DATA Banner:** Must remain visible throughout the app to clearly communicate that this is a simulated student prototype.

---

## 4. How to Run the Project (Step-by-Step Commands)

```powershell
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Generate the dataset (3,000 trips across 60 days)
python src/generate_data.py

# Step 3: Train all four machine learning models
python src/train.py

# Step 4: Run the automated verification test suite
python test_system.py

# Step 5: Start the Streamlit web dashboard
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 5. Quick Viva / Interview Answers for Mentors

- **Q: Why is your dataset split chronologically instead of randomly?**  
  *A: Transport data is a time series. Randomly shuffling would leak future weather and traffic trends into past training data. Chronological splitting ensures the model is always tested on unseen future days.*

- **Q: Why did Linear Regression win over Random Forest on this dataset?**  
  *A: Journey duration has a strong linear relationship with distance and traffic index. Linear Regression achieved a Validation MAE of 2.77 minutes, compared to 3.09 minutes for Random Forest and 4.04 minutes for Decision Tree.*

- **Q: What is the difference between MAE and RMSE?**  
  *A: MAE is the average absolute error in minutes (easy for dispatchers to understand). RMSE squares errors before averaging, penalizing occasional large prediction errors.*

- **Q: How is data leakage prevented in this system?**  
  *A: The prediction function only accepts inputs available at the terminal departure moment (distance, traffic, rain, stops, departure hour, weekday). Ground-truth actual duration and the late flag are strictly excluded from inputs.*

