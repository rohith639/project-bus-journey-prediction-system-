"""
Model Training and Evaluation Script
Bus Journey Time and Delay Prediction System - Tamil Nadu Transport
Author: AIML Student Internship Project (Career Solutions)

SIMULATED DATA - NOT A LIVE TRANSPORT FEED
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Scikit-learn tools for preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Four required machine learning algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics for evaluating regression and classification
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def inspect_dataset(df):
    """Simple function to print dataset summary and check data quality."""
    print("==========================================")
    print("STEP 1: DATA INSPECTION")
    print("==========================================")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values Check:")
    print(df.isnull().sum())

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    # Check for invalid values
    invalid_distance = (df['route_distance_km'] <= 0).sum()
    invalid_duration = (df['actual_duration_min'] <= 0).sum()
    invalid_rain = (df['rain_mm_last_hour'] < 0).sum()
    invalid_traffic = ((df['departure_traffic_index'] < 0) | (df['departure_traffic_index'] > 1)).sum()
    print(f"Invalid Values: distance={invalid_distance}, duration={invalid_duration}, rain={invalid_rain}, traffic={invalid_traffic}")

    # Outlier detection using the IQR method on journey duration
    q1 = df['actual_duration_min'].quantile(0.25)
    q3 = df['actual_duration_min'].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = df[(df['actual_duration_min'] < lower_limit) | (df['actual_duration_min'] > upper_limit)]
    print(f"\nIQR Outlier Detection: Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"Acceptable range: [{lower_limit:.1f}, {upper_limit:.1f}]")
    print(f"Outliers found: {len(outliers)} rows")
    print("==========================================")

def train_and_evaluate():
    # Folder paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "trips.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load the historical dataset
    df = pd.read_csv(data_file)
    inspect_dataset(df)

    # 2. Chronological Split (60% Train, 20% Validation, 20% Test)
    # We sort unique dates to preserve time order (NO random shuffling)
    unique_dates = sorted(df['departure_date'].unique())
    total_dates = len(unique_dates)

    train_end = int(0.60 * total_dates)
    val_end = int(0.80 * total_dates)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df['departure_date'].isin(train_dates)].copy()
    val_df = df[df['departure_date'].isin(val_dates)].copy()
    test_df = df[df['departure_date'].isin(test_dates)].copy()

    print(f"\nChronological Split:")
    print(f"Training:   {len(train_df)} trips ({len(train_dates)} days)")
    print(f"Validation: {len(val_df)} trips ({len(val_dates)} days)")
    print(f"Testing:    {len(test_df)} trips ({len(test_dates)} days)")

    # 3. Features available at departure time (NO LEAKAGE)
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
    categorical_features = ['route_id', 'weekday']
    numerical_features = [
        'route_distance_km',
        'scheduled_duration_min',
        'departure_hour',
        'stop_count',
        'departure_traffic_index',
        'rain_mm_last_hour'
    ]

    # Target variables
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

    # 4. Preprocessing: Fit ONLY on training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    preprocessor.fit(X_train)

    # Transform features
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save fitted preprocessor
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))

    # 5. Baseline Calculations on Validation Data
    # Regression baseline: predicting scheduled_duration_min
    baseline_val_preds = val_df['scheduled_duration_min']
    baseline_mae = mean_absolute_error(y_val_duration, baseline_val_preds)
    baseline_rmse = root_mean_squared_error(y_val_duration, baseline_val_preds)

    # Classification baseline: predicting most frequent class (always 0 = on time)
    baseline_clf_accuracy = (y_val_late == 0).mean()

    print("\n==========================================")
    print("STEP 2: BASELINE BENCHMARKS")
    print("==========================================")
    print(f"Scheduled Baseline MAE:  {baseline_mae:.2f} min, RMSE: {baseline_rmse:.2f} min")
    print(f"Majority Baseline Accuracy: {baseline_clf_accuracy*100:.2f}%")

    # 6. Train the 4 Required Machine Learning Algorithms
    print("\n==========================================")
    print("STEP 3: TRAINING 4 MACHINE LEARNING ALGORITHMS")
    print("==========================================")

    # Algorithm 1: Linear Regression (Predicts journey duration)
    model_lr = LinearRegression()
    model_lr.fit(X_train_proc, y_train_duration)

    # Algorithm 2: Logistic Regression (Predicts late probability & status)
    model_log = LogisticRegression(random_state=42, max_iter=1000)
    model_log.fit(X_train_proc, y_train_late)

    # Algorithm 3: Decision Tree Regressor (Predicts journey duration)
    model_dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    model_dt.fit(X_train_proc, y_train_duration)

    # Algorithm 4: Random Forest Regressor (Predicts journey duration)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model_rf.fit(X_train_proc, y_train_duration)

    # Save all four initial models
    joblib.dump(model_lr, os.path.join(models_dir, "linear_regression.pkl"))
    joblib.dump(model_log, os.path.join(models_dir, "logistic_regression.pkl"))
    joblib.dump(model_dt, os.path.join(models_dir, "decision_tree.pkl"))
    joblib.dump(model_rf, os.path.join(models_dir, "random_forest.pkl"))

    # 7. Evaluate Regression Models on Validation Set
    val_results = []

    # Baseline
    val_results.append({
        "Model": "Baseline (Scheduled Duration)",
        "Type": "Baseline",
        "MAE_min": round(baseline_mae, 3),
        "RMSE_min": round(baseline_rmse, 3)
    })

    # Linear Regression
    lr_preds = model_lr.predict(X_val_proc)
    val_results.append({
        "Model": "Linear Regression",
        "Type": "Regression",
        "MAE_min": round(mean_absolute_error(y_val_duration, lr_preds), 3),
        "RMSE_min": round(root_mean_squared_error(y_val_duration, lr_preds), 3)
    })

    # Decision Tree
    dt_preds = model_dt.predict(X_val_proc)
    val_results.append({
        "Model": "Decision Tree Regressor",
        "Type": "Regression",
        "MAE_min": round(mean_absolute_error(y_val_duration, dt_preds), 3),
        "RMSE_min": round(root_mean_squared_error(y_val_duration, dt_preds), 3)
    })

    # Random Forest
    rf_preds = model_rf.predict(X_val_proc)
    val_results.append({
        "Model": "Random Forest Regressor",
        "Type": "Regression",
        "MAE_min": round(mean_absolute_error(y_val_duration, rf_preds), 3),
        "RMSE_min": round(root_mean_squared_error(y_val_duration, rf_preds), 3)
    })

    val_df_report = pd.DataFrame(val_results)
    val_report_path = os.path.join(reports_dir, "validation_metrics.csv")
    val_df_report.to_csv(val_report_path, index=False)

    print("\nValidation Metrics for Regression Models:")
    print(val_df_report.to_string(index=False))

    # Evaluate Logistic Regression on Validation Set
    log_val_preds = model_log.predict(X_val_proc)
    clf_accuracy = accuracy_score(y_val_late, log_val_preds)
    clf_precision = precision_score(y_val_late, log_val_preds, zero_division=0)
    clf_recall = recall_score(y_val_late, log_val_preds, zero_division=0)
    clf_f1 = f1_score(y_val_late, log_val_preds, zero_division=0)
    clf_cm = confusion_matrix(y_val_late, log_val_preds)

    print("\nLogistic Regression Validation Metrics:")
    print(f"Accuracy:  {clf_accuracy*100:.2f}% (vs Baseline: {baseline_clf_accuracy*100:.2f}%)")
    print(f"Precision: {clf_precision:.4f}")
    print(f"Recall:    {clf_recall:.4f}")
    print(f"F1-Score:  {clf_f1:.4f}")
    print(f"Confusion Matrix:\n{clf_cm}")

    # 8. Select Best Regression Model based on lowest Validation MAE
    reg_models_only = val_df_report[val_df_report["Type"] == "Regression"]
    best_row = reg_models_only.sort_values(by="MAE_min").iloc[0]
    best_model_name = best_row["Model"]
    print(f"\n--> Selected Best Model: {best_model_name} (Lowest MAE: {best_row['MAE_min']} min)")

    # 9. Retrain Selected Model on Train + Validation Data (80%)
    print("\n==========================================")
    print("STEP 4: RETRAINING SELECTED MODEL ON 80% DATA")
    print("==========================================")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[features]
    y_train_val_duration = train_val_df[target_duration]
    y_train_val_late = train_val_df[target_late]

    # Re-fit preprocessor on 80% data
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    final_preprocessor.fit(X_train_val)
    X_train_val_proc = final_preprocessor.transform(X_train_val)
    X_test_proc_final = final_preprocessor.transform(X_test)

    # Initialize and train fresh instance of winning model
    if best_model_name == "Linear Regression":
        final_reg_model = LinearRegression()
    elif best_model_name == "Decision Tree Regressor":
        final_reg_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    else:
        final_reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)

    final_reg_model.fit(X_train_val_proc, y_train_val_duration)

    # Also retrain Logistic Regression on 80% data
    final_clf_model = LogisticRegression(random_state=42, max_iter=1000)
    final_clf_model.fit(X_train_val_proc, y_train_val_late)

    # Save final retrained models and metadata
    joblib.dump(final_reg_model, os.path.join(models_dir, "best_regression_model.pkl"))
    joblib.dump(final_clf_model, os.path.join(models_dir, "final_logistic_regression.pkl"))
    joblib.dump(final_preprocessor, os.path.join(models_dir, "preprocessor.pkl"))

    model_metadata = {
        "best_regression_model_name": best_model_name,
        "validation_mae": float(best_row["MAE_min"]),
        "validation_rmse": float(best_row["RMSE_min"]),
        "features": features,
        "model_version": "v1.0-tn-prod"
    }
    with open(os.path.join(models_dir, "selected_model_info.json"), "w") as f:
        json.dump(model_metadata, f, indent=4)

    # 10. Final Evaluation on Untouched Test Set (20%)
    print("\n==========================================")
    print("STEP 5: EVALUATION ON UNTOUCHED TEST SET")
    print("==========================================")

    test_preds_duration = final_reg_model.predict(X_test_proc_final)
    test_mae = mean_absolute_error(y_test_duration, test_preds_duration)
    test_rmse = root_mean_squared_error(y_test_duration, test_preds_duration)

    test_baseline_preds = test_df['scheduled_duration_min']
    test_base_mae = mean_absolute_error(y_test_duration, test_baseline_preds)
    test_base_rmse = root_mean_squared_error(y_test_duration, test_baseline_preds)

    test_preds_late = final_clf_model.predict(X_test_proc_final)
    test_accuracy = accuracy_score(y_test_late, test_preds_late)
    test_precision = precision_score(y_test_late, test_preds_late, zero_division=0)
    test_recall = recall_score(y_test_late, test_preds_late, zero_division=0)
    test_f1 = f1_score(y_test_late, test_preds_late, zero_division=0)
    test_base_acc = (y_test_late == 0).mean()

    test_report_rows = [
        {
            "Metric": "Regression MAE (min)",
            "Model_Result": round(test_mae, 3),
            "Baseline_Result": round(test_base_mae, 3),
            "Improvement": f"{((test_base_mae - test_mae) / test_base_mae) * 100:.1f}%"
        },
        {
            "Metric": "Regression RMSE (min)",
            "Model_Result": round(test_rmse, 3),
            "Baseline_Result": round(test_base_rmse, 3),
            "Improvement": f"{((test_base_rmse - test_rmse) / test_base_rmse) * 100:.1f}%"
        },
        {
            "Metric": "Classification Accuracy",
            "Model_Result": f"{test_accuracy * 100:.2f}%",
            "Baseline_Result": f"{test_base_acc * 100:.2f}%",
            "Improvement": f"{(test_accuracy - test_base_acc) * 100:+.2f}%"
        },
        {
            "Metric": "Classification F1-Score",
            "Model_Result": round(test_f1, 4),
            "Baseline_Result": "0.0000",
            "Improvement": f"+{test_f1:.4f}"
        },
        {
            "Metric": "Classification Precision",
            "Model_Result": round(test_precision, 4),
            "Baseline_Result": "0.0000",
            "Improvement": f"+{test_precision:.4f}"
        },
        {
            "Metric": "Classification Recall",
            "Model_Result": round(test_recall, 4),
            "Baseline_Result": "0.0000",
            "Improvement": f"+{test_recall:.4f}"
        }
    ]

    test_df_report = pd.DataFrame(test_report_rows)
    test_report_path = os.path.join(reports_dir, "test_metrics.csv")
    test_df_report.to_csv(test_report_path, index=False)

    print("\nFinal Test Metrics:")
    print(test_df_report.to_string(index=False))
    print(f"\nAll models and reports saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
