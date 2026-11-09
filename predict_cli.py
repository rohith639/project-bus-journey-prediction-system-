"""
Interactive Terminal Prediction Tool for Bus Journey Time & Delay
Author: AIML Student Internship Project (Career Solutions)

SIMULATED DATA – NOT A LIVE TRANSPORT FEED
Allows testing the prediction pipeline directly from the command line.
"""

from datetime import datetime
from src.predict import predict_bus_journey, KNOWN_ROUTES
from src.database import log_trip_event_and_prediction
import json

def run_cli():
    print("=" * 65)
    print("🚌 BUS JOURNEY TIME & DELAY PREDICTOR – TAMIL NADU TRANSPORT")
    print("   SIMULATED DATA – NOT A LIVE TRANSPORT FEED")
    print("=" * 65)

    print("\nAvailable Fictional Routes:")
    routes_list = list(KNOWN_ROUTES.keys())
    for idx, r_id in enumerate(routes_list, 1):
        print(f"  [{idx}] {r_id}: {KNOWN_ROUTES[r_id]}")

    print("\n--- Quick Demo Prediction ---")
    route_choice = "TN-DEMO-01"
    distance = 138.0
    scheduled_duration = 180.0
    stops = 10
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Selected Route:       {route_choice} (Chennai -> Vellore)")
    print(f"Route Distance:       {distance} km")
    print(f"Scheduled Duration:   {scheduled_duration} minutes")
    print(f"Stop Count:           {stops}")
    print(f"Departure Time:       {now_str}")

    # Simulated peak rush-hour traffic & rain
    traffic_index = 0.65
    rain_mm = 8.0
    print(f"Traffic Congestion:   {traffic_index} (Heavy flow)")
    print(f"Rainfall Last Hour:   {rain_mm} mm (Light shower)")

    print("\nRunning Machine Learning Inference...")
    result = predict_bus_journey(
        route_id=route_choice,
        route_distance_km=distance,
        scheduled_duration_min=scheduled_duration,
        departure_timestamp=now_str,
        stop_count=stops,
        departure_traffic_index=traffic_index,
        rain_mm_last_hour=rain_mm,
        source="CLI Terminal Demo"
    )

    if result["status"] == "SUCCESS":
        pred_duration = result["predicted_duration_min"]
        arrival = result["predicted_arrival_time"]
        late_prob = result["late_probability"]
        late_label = result["late_label"]

        print("\n" + "=" * 45)
        print("🎯 PREDICTION RESULTS:")
        print("=" * 45)
        print(f"  Predicted Duration:   {pred_duration} minutes")
        print(f"  Estimated Arrival:    {arrival}")
        print(f"  Late Probability:     {late_prob * 100:.1f}%")
        print(f"  Delay Status:         [{late_label.upper()}]")
        print(f"  Regression Model:     {result['selected_regression_model']}")
        print(f"  Model Version:        {result['model_version']}")
        print("=" * 45)

        # Log prediction to SQLite
        payload = json.dumps({"distance": distance, "traffic": traffic_index, "rain": rain_mm})
        log_trip_event_and_prediction(
            trip_id=f"CLI_{datetime.now().strftime('%H%M%S')}",
            route_id=route_choice,
            departure_timestamp=now_str,
            input_data_json=payload,
            source="CLI Terminal Demo",
            predicted_duration_min=pred_duration,
            predicted_arrival=arrival,
            late_probability=late_prob,
            late_label=late_label,
            model_version=result["model_version"],
            predicted_at=result["prediction_timestamp"]
        )
        print("✓ Logged prediction to SQLite database (bus_predictions.db)\n")
    else:
        print(f"❌ Error: {result.get('error_message')}")

if __name__ == "__main__":
    run_cli()

