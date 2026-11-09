"""
Prediction Pipeline for Bus Journey Time and Delay Prediction
Bus Journey Time and Delay Prediction System - Tamil Nadu Transport
Author: AIML Student Internship Project (Career Solutions)

SIMULATED DATA - NOT A LIVE TRANSPORT FEED
"""

import os
import json
import joblib
from datetime import datetime, timedelta
import pandas as pd

# Base folder paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 5 Known Tamil Nadu Routes in the simulation
KNOWN_ROUTES = {
    "TN-DEMO-01": "Chennai (CMBT) -> Vellore (138 km)",
    "TN-DEMO-02": "Coimbatore -> Salem (164 km)",
    "TN-DEMO-03": "Madurai -> Trichy (132 km)",
    "TN-DEMO-04": "Tirunelveli -> Madurai (158 km)",
    "TN-DEMO-05": "Chennai (CMBT) -> Villupuram (152 km)"
}

def load_models():
    """Simple function to load saved models and preprocessor from disk."""
    reg_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    clf_path = os.path.join(MODELS_DIR, "final_logistic_regression.pkl")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    meta_path = os.path.join(MODELS_DIR, "selected_model_info.json")

    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)
    preprocessor = joblib.load(prep_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return reg_model, clf_model, preprocessor, metadata

def validate_inputs(route_id, distance_km, scheduled_min, stops, traffic_index, rain_mm):
    """
    Validates user input to prevent negative or impossible numbers.
    Returns: (is_valid, error_message, warning_message)
    """
    if distance_km is None or scheduled_min is None or traffic_index is None:
        return False, "Missing required inputs (distance, scheduled time or traffic index).", None

    if distance_km <= 0:
        return False, f"Invalid distance ({distance_km} km). Distance must be greater than 0.", None

    if scheduled_min <= 0:
        return False, f"Invalid scheduled time ({scheduled_min} min). Must be greater than 0.", None

    if rain_mm < 0:
        return False, f"Invalid rainfall ({rain_mm} mm). Rainfall cannot be negative.", None

    if traffic_index < 0.0 or traffic_index > 1.0:
        return False, f"Traffic index ({traffic_index}) must be between 0.0 and 1.0.", None

    if stops < 0:
        return False, f"Invalid stop count ({stops}). Stops cannot be negative.", None

    warning = None
    if route_id not in KNOWN_ROUTES:
        warning = f"Unknown route '{route_id}'! Flagged for manual review."

    return True, None, warning

def predict_bus_journey(
    route_id: str,
    route_distance_km: float,
    scheduled_duration_min: float,
    departure_timestamp: str,
    stop_count: int,
    departure_traffic_index: float,
    rain_mm_last_hour: float,
    source: str = "Manual Input",
    check_staleness: bool = False,
    staleness_threshold_sec: int = 120
):
    """
    Predicts journey duration and late probability using strictly departure-time data.
    """
    # 1. Check for invalid inputs
    is_valid, error_msg, warning_msg = validate_inputs(
        route_id=route_id,
        distance_km=route_distance_km,
        scheduled_min=scheduled_duration_min,
        stops=stop_count,
        traffic_index=departure_traffic_index,
        rain_mm=rain_mm_last_hour
    )
    if not is_valid:
        return {
            "status": "ERROR",
            "error_message": error_msg,
            "data_source": source
        }

    # 2. Parse departure time string and extract hour and weekday
    try:
        if isinstance(departure_timestamp, str):
            if len(departure_timestamp) == 16:
                dt_dep = datetime.strptime(departure_timestamp, "%Y-%m-%d %H:%M")
            else:
                dt_dep = datetime.strptime(departure_timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dt_dep = departure_timestamp
    except Exception:
        return {
            "status": "ERROR",
            "error_message": "Invalid date/time format. Please use YYYY-MM-DD HH:MM:SS.",
            "data_source": source
        }

    dep_hour = dt_dep.hour
    weekday_name = dt_dep.strftime("%A")
    prediction_time = datetime.now()

    # 3. Check for stale feeds (if enabled, feeds older than 120 seconds fallback to schedule)
    if check_staleness:
        age_seconds = abs((prediction_time - dt_dep).total_seconds())
        if age_seconds > staleness_threshold_sec:
            fallback_arrival = dt_dep + timedelta(minutes=float(scheduled_duration_min))
            return {
                "status": "SCHEDULED_FALLBACK",
                "fallback_reason": f"Feed timestamp is {int(age_seconds)}s old (> {staleness_threshold_sec}s). Fallback to scheduled time.",
                "predicted_duration_min": round(float(scheduled_duration_min), 1),
                "predicted_arrival_time": fallback_arrival.strftime("%Y-%m-%d %H:%M:%S"),
                "late_probability": 0.0,
                "late_label": "Scheduled (Fallback)",
                "selected_regression_model": "Fallback (Scheduled Timetable)",
                "model_version": "v1.0-fallback",
                "prediction_timestamp": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": source,
                "warning": warning_msg,
                "is_fallback": True
            }

    # 4. Load trained models
    reg_model, clf_model, preprocessor, metadata = load_models()
    model_version = metadata.get("model_version", "v1.0-tn-prod")
    model_name = metadata.get("best_regression_model_name", "Linear Regression")

    # 5. Create DataFrame with departure-time features only (NO LEAKAGE)
    input_data = pd.DataFrame([{
        "route_id": route_id,
        "route_distance_km": float(route_distance_km),
        "scheduled_duration_min": float(scheduled_duration_min),
        "departure_hour": int(dep_hour),
        "weekday": weekday_name,
        "stop_count": int(stop_count),
        "departure_traffic_index": float(departure_traffic_index),
        "rain_mm_last_hour": float(rain_mm_last_hour)
    }])

    # Preprocess features using fitted pipeline
    input_proc = preprocessor.transform(input_data)

    # 6. Predict duration using the best regression model
    predicted_duration_raw = reg_model.predict(input_proc)[0]
    # Keep duration realistic (at least 70% of scheduled)
    predicted_duration = round(max(float(scheduled_duration_min) * 0.70, float(predicted_duration_raw)), 1)

    # 7. Predict late probability using Logistic Regression
    late_prob = float(clf_model.predict_proba(input_proc)[0][1])
    # Threshold at 0.50 (50%)
    if late_prob >= 0.50:
        late_label = "Late"
    else:
        late_label = "On Time"

    # 8. Calculate estimated arrival time
    predicted_arrival = dt_dep + timedelta(minutes=predicted_duration)

    return {
        "status": "SUCCESS",
        "route_id": route_id,
        "departure_time": dt_dep.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_duration_min": predicted_duration,
        "predicted_arrival_time": predicted_arrival.strftime("%Y-%m-%d %H:%M:%S"),
        "late_probability": round(late_prob, 4),
        "late_label": late_label,
        "selected_regression_model": model_name,
        "model_version": model_version,
        "prediction_timestamp": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": source,
        "warning": warning_msg,
        "is_fallback": False
    }

if __name__ == "__main__":
    print("Testing simplified predict_bus_journey...")
    test_result = predict_bus_journey(
        route_id="TN-DEMO-01",
        route_distance_km=138.0,
        scheduled_duration_min=180.0,
        departure_timestamp="2026-03-02 08:30:00",
        stop_count=10,
        departure_traffic_index=0.75,
        rain_mm_last_hour=15.0
    )
    print("Result:")
    for key, value in test_result.items():
        print(f"  {key}: {value}")
