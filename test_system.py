"""
Comprehensive Verification and Test Script for Bus Journey Prediction System
Author: AIML Student Internship Project (Career Solutions)
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np

from src.predict import predict_bus_journey, validate_inputs, KNOWN_ROUTES
from src.database import (
    init_db,
    log_trip_event_and_prediction,
    get_prediction_history,
    get_prediction_count
)

def run_tests():
    print("=" * 60)
    print("STARTING COMPLETE SYSTEM VERIFICATION TESTS")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    # TEST 1 & 2: Dataset files exist
    trips_path = os.path.join(data_dir, "trips.csv")
    events_path = os.path.join(data_dir, "departure_events.csv")

    assert os.path.exists(trips_path), "trips.csv does not exist!"
    assert os.path.exists(events_path), "departure_events.csv does not exist!"
    print("✓ [TEST 1 & 2] Both CSV files exist on disk.")

    # TEST 3 & 4: Dataset records and columns
    df_trips = pd.read_csv(trips_path)
    df_events = pd.read_csv(events_path)

    expected_trips_cols = [
        "trip_id", "route_id", "departure_date", "route_distance_km",
        "scheduled_duration_min", "departure_hour", "weekday",
        "stop_count", "departure_traffic_index", "rain_mm_last_hour",
        "actual_duration_min", "late_flag"
    ]
    for col in expected_trips_cols:
        assert col in df_trips.columns, f"Missing required column in trips.csv: {col}"

    assert 2900 <= len(df_trips) <= 3100, f"Expected ~3,000 records, got {len(df_trips)}"
    assert len(df_events) == len(df_trips), "Events count should match trips count"

    # Verify no data leakage in departure_events.csv
    assert "actual_duration_min" not in df_events.columns, "Leakage! actual_duration_min in events CSV!"
    assert "late_flag" not in df_events.columns, "Leakage! late_flag in events CSV!"
    print(f"✓ [TEST 3 & 4] Dataset verified: {len(df_trips)} rows, all 12 columns present, zero leakage in departure_events.csv.")

    # TEST 5 & 6: Models exist
    required_models = [
        "linear_regression.pkl",
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "random_forest.pkl",
        "best_regression_model.pkl",
        "preprocessor.pkl",
        "selected_model_info.json"
    ]
    for m in required_models:
        path = os.path.join(models_dir, m)
        assert os.path.exists(path), f"Missing model artifact: {m}"
    print("✓ [TEST 5 & 6] All 4 ML models + best retrained model + preprocessor verified in models/.")

    # TEST 7 & 8: Reports exist
    val_rep = os.path.join(reports_dir, "validation_metrics.csv")
    test_rep = os.path.join(reports_dir, "test_metrics.csv")
    assert os.path.exists(val_rep), "validation_metrics.csv missing!"
    assert os.path.exists(test_rep), "test_metrics.csv missing!"
    df_val = pd.read_csv(val_rep)
    df_test = pd.read_csv(test_rep)
    print("✓ [TEST 7 & 8] Evaluation reports verified:")
    print(df_val.to_string(index=False))
    print(df_test.to_string(index=False))

    # TEST 9: Prediction function works
    pred = predict_bus_journey(
        route_id="TN-DEMO-02",
        route_distance_km=164.0,
        scheduled_duration_min=210.0,
        departure_timestamp="2026-03-05 09:15:00",
        stop_count=14,
        departure_traffic_index=0.60,
        rain_mm_last_hour=5.0,
        source="Test Automation"
    )
    assert pred["status"] == "SUCCESS", f"Prediction failed: {pred}"
    assert "predicted_duration_min" in pred
    assert "predicted_arrival_time" in pred
    assert "late_probability" in pred
    assert pred["late_label"] in ["Late", "On Time"]
    print(f"✓ [TEST 9] Prediction function verified: Duration = {pred['predicted_duration_min']} min, Status = {pred['late_label']}, Prob = {pred['late_probability']}")

    # TEST 10: SQLite Database operations
    init_db()
    count_before = get_prediction_count()
    test_trip_id = "TEST_TRIP_VERIFY_01"
    log_trip_event_and_prediction(
        trip_id=test_trip_id,
        route_id="TN-DEMO-02",
        departure_timestamp="2026-03-05 09:15:00",
        input_data_json=json.dumps({"test": True}),
        source="Test Automation",
        predicted_duration_min=pred["predicted_duration_min"],
        predicted_arrival=pred["predicted_arrival_time"],
        late_probability=pred["late_probability"],
        late_label=pred["late_label"],
        model_version=pred["model_version"],
        predicted_at=pred["prediction_timestamp"]
    )
    count_after = get_prediction_count()
    assert count_after == count_before + 1, "Database insertion count mismatch!"
    history_df = get_prediction_history(limit=5)
    assert not history_df.empty, "Database history is empty!"
    assert history_df.iloc[0]["trip_id"] == test_trip_id, "Latest logged trip ID mismatch!"
    print(f"✓ [TEST 10] SQLite logging and retrieval verified ({count_after} total records in DB).")

    # TEST 14 & 15: Input Validation & Safety checks
    # Test negative distance
    neg_dist = predict_bus_journey(
        route_id="TN-DEMO-01", route_distance_km=-10.0, scheduled_duration_min=180.0,
        departure_timestamp="2026-03-05 09:00:00", stop_count=10,
        departure_traffic_index=0.3, rain_mm_last_hour=0.0
    )
    assert neg_dist["status"] == "ERROR", "Failed to reject negative distance!"

    # Test negative rainfall
    neg_rain = predict_bus_journey(
        route_id="TN-DEMO-01", route_distance_km=138.0, scheduled_duration_min=180.0,
        departure_timestamp="2026-03-05 09:00:00", stop_count=10,
        departure_traffic_index=0.3, rain_mm_last_hour=-5.0
    )
    assert neg_rain["status"] == "ERROR", "Failed to reject negative rainfall!"

    # Test traffic index > 1.0
    inv_traffic = predict_bus_journey(
        route_id="TN-DEMO-01", route_distance_km=138.0, scheduled_duration_min=180.0,
        departure_timestamp="2026-03-05 09:00:00", stop_count=10,
        departure_traffic_index=1.5, rain_mm_last_hour=0.0
    )
    assert inv_traffic["status"] == "ERROR", "Failed to reject invalid traffic index!"

    # Test unknown route warning
    unk_route = predict_bus_journey(
        route_id="UNKNOWN-99", route_distance_km=100.0, scheduled_duration_min=120.0,
        departure_timestamp="2026-03-05 09:00:00", stop_count=5,
        departure_traffic_index=0.4, rain_mm_last_hour=0.0
    )
    assert unk_route["status"] == "SUCCESS"
    assert "Unknown route" in unk_route["warning"]
    print("✓ [TEST 14 & 15] Input validation and safety checks verified (Negative values rejected, unknown routes flagged).")

    # TEST 13: Departure Replay Event Step
    sample_replay_row = df_events.iloc[0]
    replay_res = predict_bus_journey(
        route_id=sample_replay_row["route_id"],
        route_distance_km=float(sample_replay_row["route_distance_km"]),
        scheduled_duration_min=float(sample_replay_row["scheduled_duration_min"]),
        departure_timestamp=str(sample_replay_row["departure_timestamp"]),
        stop_count=int(sample_replay_row["stop_count"]),
        departure_traffic_index=float(sample_replay_row["departure_traffic_index"]),
        rain_mm_last_hour=float(sample_replay_row["rain_mm_last_hour"]),
        source="Departure Replay Verification"
    )
    assert replay_res["status"] == "SUCCESS", "Departure replay step failed!"
    print("✓ [TEST 13] Departure replay pipeline verified on event 0.")

    print("\n" + "=" * 60)
    print("ALL SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

