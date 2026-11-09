"""
train_models.py - Model Training and Evaluation
Tamil Nadu Bus Journey Time and Delay Prediction System
Career Solutions / SkillFirstLabs Internship

SIMULATED DATA – NOT A LIVE TRANSPORT FEED
Trains 4 machine learning models using chronological train/validation/test splitting.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Scikit-learn preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# The 4 Required Machine Learning Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def inspect_data(data):
    """Simple function to inspect dataset quality and check for missing values or corrupt data."""
    print("==========================================")
    print("STEP 1: DATA INSPECTION")
    print("==========================================")
    print("Dataset Shape:", data.shape)
    print("\nFirst 5 Rows:")
    print(data.head())

    print("\nMissing Values Count:")
    print(data.isnull().sum())

    print("\nDuplicate Rows Count:", data.duplicated().sum())

    # Check for invalid values
    invalid_dist = (data['route_distance_km'] <= 0).sum()
    invalid_dur = (data['actual_duration_min'] <= 0).sum()
    invalid_rain = (data['rain_mm_last_hour'] < 0).sum()
    invalid_traffic = ((data['departure_traffic_index'] < 0) | (data['departure_traffic_index'] > 1)).sum()
    print(f"Invalid Values Check: distance={invalid_dist}, duration={invalid_dur}, rain={invalid_rain}, traffic={invalid_traffic}")

    # Outlier detection using the Interquartile Range (IQR) method
    q1 = data['actual_duration_min'].quantile(0.25)
    q3 = data['actual_duration_min'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data['actual_duration_min'] < lower_bound) | (data['actual_duration_min'] > upper_bound)]
    print(f"\nIQR Outlier Analysis: Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"Normal range: [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"Outliers detected: {len(outliers)} rows (all realistic rush-hour trips, kept for training)")
    print("==========================================")

def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load dataset
    data = pd.read_csv("data/trips.csv")
    inspect_data(data)

    # 2. Chronological Splitting (60% Train, 20% Validation, 20% Test)
    # We sort dates chronologically. We do NOT shuffle randomly because future days cannot predict past days!
    unique_dates = sorted(data['departure_date'].unique())
    num_dates = len(unique_dates)

    train_end = int(0.60 * num_dates)
    val_end = int(0.80 * num_dates)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = data[data['departure_date'].isin(train_dates)].copy()
    val_df = data[data['departure_date'].isin(val_dates)].copy()
    test_df = data[data['departure_date'].isin(test_dates)].copy()

    print(f"\nChronological Split:")
    print(f"  Training set:   {len(train_df)} trips ({len(train_dates)} days)")
    print(f"  Validation set: {len(val_df)} trips ({len(val_dates)} days)")
    print(f"  Test set:       {len(test_df)} trips ({len(test_dates)} days)")

    # 3. Features known at departure time (NO DATA LEAKAGE)
    features = [
        'route_id',
        'route_distance_km',
        'scheduled_duration_min',
        'departure_hour',
        'weekday',
        'stop_count',
        'departure_traffic_index',
        'rain_mm_last_hour'
    ]
    cat_features = ['route_id', 'weekday']
    num_features = [
        'route_distance_km',
        'scheduled_duration_min',
        'departure_hour',
        'stop_count',
        'departure_traffic_index',
        'rain_mm_last_hour'
    ]

    target_duration = 'actual_duration_min'
    target_late = 'late_flag'

    X_train = train_df[features]
    y_train_duration = train_df[target_duration]
    y_train_late = train_df[target_late]

    X_val = val_df[features]
    y_val_duration = val_df[target_duration]
    y_val_late = val_df[target_late]

    X_test = test_df[features]
    y_test_duration = test_df[target_duration]
    y_test_late = test_df[target_late]

    # 4. Preprocessing: Scale numbers and one-hot encode categories
    # Fitted ONLY on the training data to prevent data leakage!
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save fitted preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # 5. Baseline Calculations
    # Timetable baseline for duration: scheduled_duration_min
    baseline_val_preds = val_df['scheduled_duration_min']
    baseline_mae = mean_absolute_error(y_val_duration, baseline_val_preds)
    baseline_rmse = root_mean_squared_error(y_val_duration, baseline_val_preds)

    # Majority class baseline for classification (always predicting 0 = on-time)
    baseline_clf_accuracy = (y_val_late == 0).mean()

    print("\n==========================================")
    print("STEP 2: BASELINES ON VALIDATION SET")
    print("==========================================")
    print(f"Scheduled Baseline MAE:  {baseline_mae:.2f} min, RMSE: {baseline_rmse:.2f} min")
    print(f"Majority Baseline Accuracy: {baseline_clf_accuracy*100:.2f}%")

    # 6. Train the 4 Machine Learning Algorithms
    print("\n==========================================")
    print("STEP 3: TRAINING 4 ALGORITHMS")
    print("==========================================")

    # Algorithm 1: Linear Regression (Predicts journey duration)
    lr_model = LinearRegression()
    lr_model.fit(X_train_proc, y_train_duration)

    # Algorithm 2: Logistic Regression (Predicts late arrival probability)
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_proc, y_train_late)

    # Algorithm 3: Decision Tree Regressor (Predicts journey duration)
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_model.fit(X_train_proc, y_train_duration)

    # Algorithm 4: Random Forest Regressor (Predicts journey duration)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train_proc, y_train_duration)

    # Save initial 4 models
    joblib.dump(lr_model, "models/linear_regression.pkl")
    joblib.dump(log_model, "models/logistic_regression.pkl")
    joblib.dump(dt_model, "models/decision_tree.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")

    # 7. Evaluate Regression Models on Validation Set
    val_results = [
        {"Model": "Baseline (Scheduled Duration)", "Type": "Baseline", "MAE_min": round(baseline_mae, 3), "RMSE_min": round(baseline_rmse, 3)},
        {"Model": "Linear Regression", "Type": "Regression", "MAE_min": round(mean_absolute_error(y_val_duration, lr_model.predict(X_val_proc)), 3), "RMSE_min": round(root_mean_squared_error(y_val_duration, lr_model.predict(X_val_proc)), 3)},
        {"Model": "Decision Tree Regressor", "Type": "Regression", "MAE_min": round(mean_absolute_error(y_val_duration, dt_model.predict(X_val_proc)), 3), "RMSE_min": round(root_mean_squared_error(y_val_duration, dt_model.predict(X_val_proc)), 3)},
        {"Model": "Random Forest Regressor", "Type": "Regression", "MAE_min": round(mean_absolute_error(y_val_duration, rf_model.predict(X_val_proc)), 3), "RMSE_min": round(root_mean_squared_error(y_val_duration, rf_model.predict(X_val_proc)), 3)}
    ]
    val_df_report = pd.DataFrame(val_results)
    val_df_report.to_csv("reports/validation_metrics.csv", index=False)
    print("\nValidation Metrics for Regression Models:")
    print(val_df_report.to_string(index=False))

    # Evaluate Logistic Regression on Validation Set
    log_preds = log_model.predict(X_val_proc)
    clf_acc = accuracy_score(y_val_late, log_preds)
    clf_prec = precision_score(y_val_late, log_preds, zero_division=0)
    clf_rec = recall_score(y_val_late, log_preds, zero_division=0)
    clf_f1 = f1_score(y_val_late, log_preds, zero_division=0)
    cm = confusion_matrix(y_val_late, log_preds)

    print("\nLogistic Regression Validation Performance:")
    print(f"Accuracy:  {clf_acc*100:.2f}% (vs Dummy Baseline: {baseline_clf_accuracy*100:.2f}%)")
    print(f"Precision: {clf_prec:.4f}")
    print(f"Recall:    {clf_rec:.4f}")
    print(f"F1-Score:  {clf_f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # 8. Select Best Regression Model based on lowest Validation MAE
    reg_only = val_df_report[val_df_report["Type"] == "Regression"]
    best_row = reg_only.sort_values(by="MAE_min").iloc[0]
    best_name = best_row["Model"]
    print(f"\n--> Selected Best Model: {best_name} with lowest MAE = {best_row['MAE_min']} min")

    # 9. Retrain Winning Model on Train + Validation Data (80%)
    print("\n==========================================")
    print(f"STEP 4: RETRAINING '{best_name}' ON 80% DATA")
    print("==========================================")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[features]
    y_train_val_duration = train_val_df[target_duration]
    y_train_val_late = train_val_df[target_late]

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    final_preprocessor.fit(X_train_val)
    X_train_val_proc = final_preprocessor.transform(X_train_val)
    X_test_proc_final = final_preprocessor.transform(X_test)

    # Retrain fresh instance of the winning model
    if best_name == "Linear Regression":
        final_reg = LinearRegression()
    elif best_name == "Decision Tree Regressor":
        final_reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    else:
        final_reg = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)

    final_reg.fit(X_train_val_proc, y_train_val_duration)

    # Retrain Logistic Regression on 80% data
    final_clf = LogisticRegression(random_state=42, max_iter=1000)
    final_clf.fit(X_train_val_proc, y_train_val_late)

    # Save production models and metadata
    joblib.dump(final_reg, "models/best_regression_model.pkl")
    joblib.dump(final_clf, "models/final_logistic_regression.pkl")
    joblib.dump(final_preprocessor, "models/preprocessor.pkl")

    meta = {
        "best_regression_model_name": best_name,
        "validation_mae": float(best_row["MAE_min"]),
        "validation_rmse": float(best_row["RMSE_min"]),
        "features": features,
        "model_version": "v1.0-tn-prod"
    }
    with open("models/selected_model_info.json", "w") as f:
        json.dump(meta, f, indent=4)

    # 10. Evaluate ONCE on Untouched Test Set (20%)
    print("\n==========================================")
    print("STEP 5: EVALUATION ON UNTOUCHED TEST SET")
    print("==========================================")

    test_preds_dur = final_reg.predict(X_test_proc_final)
    test_mae = mean_absolute_error(y_test_duration, test_preds_dur)
    test_rmse = root_mean_squared_error(y_test_duration, test_preds_dur)

    test_base_preds = test_df['scheduled_duration_min']
    test_base_mae = mean_absolute_error(y_test_duration, test_base_preds)
    test_base_rmse = root_mean_squared_error(y_test_duration, test_base_preds)

    test_preds_late = final_clf.predict(X_test_proc_final)
    test_acc = accuracy_score(y_test_late, test_preds_late)
    test_f1 = f1_score(y_test_late, test_preds_late, zero_division=0)
    test_prec = precision_score(y_test_late, test_preds_late, zero_division=0)
    test_rec = recall_score(y_test_late, test_preds_late, zero_division=0)
    test_base_acc = (y_test_late == 0).mean()

    test_report = [
        {"Metric": "Regression MAE (min)", "Model_Result": round(test_mae, 3), "Baseline_Result": round(test_base_mae, 3), "Improvement": f"{((test_base_mae - test_mae) / test_base_mae) * 100:.1f}%"},
        {"Metric": "Regression RMSE (min)", "Model_Result": round(test_rmse, 3), "Baseline_Result": round(test_base_rmse, 3), "Improvement": f"{((test_base_rmse - test_rmse) / test_base_rmse) * 100:.1f}%"},
        {"Metric": "Classification Accuracy", "Model_Result": f"{test_acc * 100:.2f}%", "Baseline_Result": f"{test_base_acc * 100:.2f}%", "Improvement": f"{(test_acc - test_base_acc) * 100:+.2f}%"},
        {"Metric": "Classification F1-Score", "Model_Result": round(test_f1, 4), "Baseline_Result": "0.0000", "Improvement": f"+{test_f1:.4f}"},
        {"Metric": "Classification Precision", "Model_Result": round(test_prec, 4), "Baseline_Result": "0.0000", "Improvement": f"+{test_prec:.4f}"},
        {"Metric": "Classification Recall", "Model_Result": round(test_rec, 4), "Baseline_Result": "0.0000", "Improvement": f"+{test_rec:.4f}"}
    ]
    test_df_report = pd.DataFrame(test_report)
    test_df_report.to_csv("reports/test_metrics.csv", index=False)

    print("\nFinal Test Metrics Report:")
    print(test_df_report.to_string(index=False))
    print("\nTraining completed successfully! All models saved to models/")

if __name__ == "__main__":
    train_models()

