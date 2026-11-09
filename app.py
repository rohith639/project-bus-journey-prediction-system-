"""
app.py - Streamlit Web Dashboard
Bus Journey Time and Delay Prediction System – Tamil Nadu Transport
Career Solutions / SkillFirstLabs Internship

SIMULATED DATA – NOT A LIVE TRANSPORT FEED
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Import simple local modules
from predict import predict_bus_journey, KNOWN_ROUTES
from database import init_db, log_prediction, get_history, get_total_count

# Page setup
st.set_page_config(
    page_title="Bus Journey Time and Delay Prediction System",
    page_icon="🚌",
    layout="wide"
)

# Initialize SQLite database
init_db()

# Route Presets for Tamil Nadu corridors
ROUTE_INFO = {
    "TN-DEMO-01": {"name": "Chennai (CMBT) -> Vellore", "distance": 138.0, "duration": 180, "stops": 10},
    "TN-DEMO-02": {"name": "Coimbatore -> Salem", "distance": 164.0, "duration": 210, "stops": 14},
    "TN-DEMO-03": {"name": "Madurai -> Trichy", "distance": 132.0, "duration": 170, "stops": 9},
    "TN-DEMO-04": {"name": "Tirunelveli -> Madurai", "distance": 158.0, "duration": 200, "stops": 12},
    "TN-DEMO-05": {"name": "Chennai (CMBT) -> Villupuram", "distance": 152.0, "duration": 195, "stops": 11}
}

# Load static files if available
df_trips = pd.read_csv("data/trips.csv") if os.path.exists("data/trips.csv") else pd.DataFrame()
df_events = pd.read_csv("data/departure_events.csv") if os.path.exists("data/departure_events.csv") else pd.DataFrame()
val_metrics = pd.read_csv("reports/validation_metrics.csv") if os.path.exists("reports/validation_metrics.csv") else pd.DataFrame()
test_metrics = pd.read_csv("reports/test_metrics.csv") if os.path.exists("reports/test_metrics.csv") else pd.DataFrame()

# Load model metadata
model_meta = {}
if os.path.exists("models/selected_model_info.json"):
    with open("models/selected_model_info.json", "r") as f:
        model_meta = json.load(f)

# Sidebar Navigation
st.sidebar.title("🚌 Bus Delay Predictor")
st.sidebar.warning("⚠️ SIMULATED DATA – NOT A LIVE TRANSPORT FEED")

section = st.sidebar.selectbox(
    "Go to Section:",
    [
        "1. Dashboard Overview",
        "2. Bus Journey Prediction",
        "3. Simulated Departure Replay",
        "4. Model Comparison",
        "5. Data Visualizations",
        "6. Prediction History"
    ]
)

st.sidebar.write("---")
st.sidebar.write("### 📥 Download Datasets")
if not df_trips.empty:
    with open("data/trips.csv", "rb") as f_trips:
        st.sidebar.download_button("Download trips.csv", f_trips, file_name="trips.csv", mime="text/csv")
if not df_events.empty:
    with open("data/departure_events.csv", "rb") as f_events:
        st.sidebar.download_button("Download departure_events.csv", f_events, file_name="departure_events.csv", mime="text/csv")

# Main Header
st.title("🚌 Bus Journey Time and Delay Prediction System")
st.write("### Tamil Nadu Transport Corridors (Simulated Prototype)")
st.warning("⚠️ **SIMULATED DATA – NOT A LIVE TRANSPORT FEED** | This application uses machine learning algorithms on simulated bus corridor data.")

# ==============================================================================
# SECTION 1: DASHBOARD OVERVIEW
# ==============================================================================
if section == "1. Dashboard Overview":
    st.header("1. Dashboard Overview")
    st.write("Welcome to the Bus Journey Time and Delay Prediction System. This system models 60 operating days across 5 fictional Tamil Nadu corridors.")

    # High level statistics
    total_records = len(df_trips)
    num_routes = df_trips['route_id'].nunique() if not df_trips.empty else 0
    late_trips = df_trips['late_flag'].sum() if not df_trips.empty else 0
    ontime_trips = total_records - late_trips

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips", f"{total_records:,}")
    col2.metric("Routes", num_routes)
    col3.metric("Late Trips (>10 min delay)", f"{late_trips:,}", f"{late_trips/total_records*100:.1f}%")
    col4.metric("On-Time Trips", f"{ontime_trips:,}", f"{ontime_trips/total_records*100:.1f}%")

    st.write("---")
    st.subheader("Local Dataset Paths:")
    st.code(f"Historical Trips CSV: {os.path.abspath('data/trips.csv')}\nDeparture Events CSV: {os.path.abspath('data/departure_events.csv')}")

    # Download buttons in overview
    d1, d2 = st.columns(2)
    with d1:
        if not df_trips.empty:
            with open("data/trips.csv", "rb") as f1:
                st.download_button("📥 Download Historical trips.csv", f1, file_name="trips.csv", mime="text/csv")
    with d2:
        if not df_events.empty:
            with open("data/departure_events.csv", "rb") as f2:
                st.download_button("📥 Download departure_events.csv", f2, file_name="departure_events.csv", mime="text/csv")

    st.write("---")
    st.subheader("Dataset Preview (First 10 Rows):")
    st.dataframe(df_trips.head(10), use_container_width=True)

    st.write("---")
    st.subheader("Data Dictionary:")
    data_dictionary = [
        {"Column": "trip_id", "Type": "Text", "Description": "Unique identifier for each simulated bus trip"},
        {"Column": "route_id", "Type": "Categorical", "Description": "Fictional route identifier (e.g., TN-DEMO-01)"},
        {"Column": "departure_date", "Type": "Date", "Description": "Date of departure (YYYY-MM-DD)"},
        {"Column": "route_distance_km", "Type": "Float", "Description": "Route corridor distance in kilometers"},
        {"Column": "scheduled_duration_min", "Type": "Integer", "Description": "Timetable scheduled journey duration in minutes"},
        {"Column": "departure_hour", "Type": "Integer", "Description": "Hour of departure (5 AM to 10 PM)"},
        {"Column": "weekday", "Type": "Categorical", "Description": "Day of the week (Monday through Sunday)"},
        {"Column": "stop_count", "Type": "Integer", "Description": "Number of scheduled intermediate stops"},
        {"Column": "departure_traffic_index", "Type": "Float", "Description": "Traffic congestion index at departure (0.0 to 1.0)"},
        {"Column": "rain_mm_last_hour", "Type": "Float", "Description": "Rainfall in millimeters recorded in the last hour"},
        {"Column": "actual_duration_min", "Type": "Float", "Description": "Actual trip duration (Target - NOT used as input)"},
        {"Column": "late_flag", "Type": "Binary (0/1)", "Description": "1 if actual duration > scheduled + 10 min, else 0"}
    ]
    st.dataframe(pd.DataFrame(data_dictionary), use_container_width=True)

# ==============================================================================
# SECTION 2: BUS JOURNEY PREDICTION
# ==============================================================================
elif section == "2. Bus Journey Prediction":
    st.header("2. Bus Journey Prediction")
    st.write("Enter departure information below. The model predicts journey duration and late probability using only departure-time features.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Departure Input Form:")

        route_id = st.selectbox(
            "Select Route:",
            list(ROUTE_INFO.keys()),
            format_func=lambda r: f"{r}: {ROUTE_INFO[r]['name']}"
        )
        preset = ROUTE_INFO[route_id]

        override = st.checkbox("Custom Distance / Stops Override", value=False)
        if override:
            dist_val = st.number_input("Distance (km):", min_value=1.0, max_value=500.0, value=float(preset["distance"]))
            sched_val = st.number_input("Scheduled Duration (min):", min_value=10.0, max_value=600.0, value=float(preset["duration"]))
            stops_val = st.number_input("Stop Count:", min_value=0, max_value=50, value=int(preset["stops"]))
        else:
            dist_val = float(preset["distance"])
            sched_val = float(preset["duration"])
            stops_val = int(preset["stops"])
            st.info(f"📍 Route Preset: {dist_val} km | {stops_val} stops | {sched_val} scheduled min")

        dep_date = st.date_input("Departure Date:", value=datetime.today())
        dep_time = st.time_input("Departure Time:", value=datetime.strptime("08:30", "%H:%M").time())
        departure_timestamp = f"{dep_date.strftime('%Y-%m-%d')} {dep_time.strftime('%H:%M:%S')}"

        traffic_index = st.slider("Departure Traffic Index (0.0=Clear, 1.0=Gridlock):", 0.0, 1.0, 0.45, 0.01)
        rain_mm = st.number_input("Rainfall in Last Hour (mm):", min_value=0.0, max_value=120.0, value=0.0, step=1.0)

        trip_id = st.text_input("Trip ID:", value=f"TRIP_{datetime.now().strftime('%H%M%S')}")

        predict_btn = st.button("🚀 Predict Journey Time & Delay", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Prediction Results:")

        if predict_btn:
            result = predict_bus_journey(
                route_id=route_id,
                route_distance_km=dist_val,
                scheduled_duration_min=sched_val,
                departure_timestamp=departure_timestamp,
                stop_count=stops_val,
                departure_traffic_index=traffic_index,
                rain_mm_last_hour=rain_mm,
                source="Dispatcher Form"
            )

            if result["status"] == "ERROR":
                st.error(f"❌ Input Validation Error: {result['error_message']}")
            else:
                pred_duration = result["predicted_duration_min"]
                arrival_time = result["predicted_arrival_time"]
                late_prob = result["late_probability"]
                late_label = result["late_label"]
                model_used = result["selected_regression_model"]
                model_ver = result["model_version"]
                pred_time = result["prediction_timestamp"]

                # Save prediction to SQLite
                input_payload = json.dumps({"distance": dist_val, "scheduled": sched_val, "traffic": traffic_index, "rain": rain_mm})
                log_prediction(
                    trip_id=trip_id,
                    route_id=route_id,
                    departure_timestamp=departure_timestamp,
                    input_data_json=input_payload,
                    source="Dispatcher Form",
                    predicted_duration_min=pred_duration,
                    predicted_arrival=arrival_time,
                    late_probability=late_prob,
                    late_label=late_label,
                    model_version=model_ver,
                    predicted_at=pred_time
                )

                # Show main metrics
                m1, m2 = st.columns(2)
                diff = round(pred_duration - sched_val, 1)
                diff_text = f"+{diff} min vs schedule" if diff >= 0 else f"{diff} min vs schedule"

                m1.metric("Predicted Duration", f"{pred_duration} min", diff_text, delta_color="inverse")
                m2.metric("Estimated Arrival", arrival_time.split(" ")[1])

                st.write("---")
                if late_label == "Late":
                    st.error(f"🚨 Predicted Status: **LATE** (Late Probability: {late_prob*100:.1f}%)")
                else:
                    st.success(f"✅ Predicted Status: **ON TIME** (Late Probability: {late_prob*100:.1f}%)")

                st.progress(late_prob, text=f"Late-Arrival Probability: {late_prob*100:.1f}%")

                st.write("---")
                st.write(f"- **Regression Model:** {model_used}")
                st.write(f"- **Classification Model:** Logistic Regression (Threshold = 0.50)")
                st.write(f"- **Model Version:** `{model_ver}`")
                st.write(f"- **Full Estimated Arrival:** `{arrival_time}`")
                st.write(f"- **Database Status:** Logged to `bus_predictions.db`")
        else:
            st.info("Fill out the departure parameters on the left and click **'Predict Journey Time & Delay'**.")

# ==============================================================================
# SECTION 3: SIMULATED DEPARTURE REPLAY
# ==============================================================================
elif section == "3. Simulated Departure Replay":
    st.header("3. Simulated Departure Replay")
    st.write("Replays simulated departure events from `data/departure_events.csv` one by one.")
    st.warning("⚠️ SIMULATED DATA – NOT A LIVE TRANSPORT FEED")

    if df_events.empty:
        st.error("No departure events found! Run `python generate_data.py` first.")
    else:
        if "replay_idx" not in st.session_state:
            st.session_state.replay_idx = 0

        # Navigation controls
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            if st.button("⬅️ Previous Event"):
                st.session_state.replay_idx = max(0, st.session_state.replay_idx - 1)
        with c2:
            if st.button("➡️ Next Event"):
                st.session_state.replay_idx = min(len(df_events) - 1, st.session_state.replay_idx + 1)
        with c3:
            if st.button("🎲 Random Event"):
                st.session_state.replay_idx = int(np.random.randint(0, len(df_events)))
        with c4:
            st.write(f"Showing Event **{st.session_state.replay_idx + 1}** of **{len(df_events):,}**")

        # 30-Second Auto-Refresh replay option
        auto_replay = st.checkbox("Enable 30-Second Auto-Refresh Replay", value=False)
        if auto_replay:
            st.write("🔄 Auto-refresh enabled: Replaying streaming events every 30 seconds.")
            import time
            time.sleep(0.5)

        event = df_events.iloc[st.session_state.replay_idx]

        st.write("---")
        st.subheader(f"Current Replayed Event: `{event['trip_id']}`")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Route", event["route_id"], ROUTE_INFO.get(event["route_id"], {}).get("name", ""))
        e2.metric("Scheduled", f"{event['scheduled_duration_min']} min", f"{event['route_distance_km']} km")
        e3.metric("Traffic Index", f"{event['departure_traffic_index']}")
        e4.metric("Rainfall", f"{event['rain_mm_last_hour']} mm")

        # Run prediction for this replayed event
        replay_result = predict_bus_journey(
            route_id=event["route_id"],
            route_distance_km=float(event["route_distance_km"]),
            scheduled_duration_min=float(event["scheduled_duration_min"]),
            departure_timestamp=str(event["departure_timestamp"]),
            stop_count=int(event["stop_count"]),
            departure_traffic_index=float(event["departure_traffic_index"]),
            rain_mm_last_hour=float(event["rain_mm_last_hour"]),
            source="Simulated Replay"
        )

        # Log prediction to SQLite
        if replay_result["status"] == "SUCCESS":
            payload = json.dumps({"distance": float(event["route_distance_km"]), "traffic": float(event["departure_traffic_index"])})
            log_prediction(
                trip_id=event["trip_id"],
                route_id=event["route_id"],
                departure_timestamp=str(event["departure_timestamp"]),
                input_data_json=payload,
                source="Simulated Replay",
                predicted_duration_min=replay_result["predicted_duration_min"],
                predicted_arrival=replay_result["predicted_arrival_time"],
                late_probability=replay_result["late_probability"],
                late_label=replay_result["late_label"],
                model_version=replay_result["model_version"],
                predicted_at=replay_result["prediction_timestamp"]
            )

        st.write("### Model Inferences for this Departure:")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Duration", f"{replay_result['predicted_duration_min']} min")
        r2.metric("Estimated Arrival", replay_result["predicted_arrival_time"])
        r3.metric("Late Probability", f"{replay_result['late_probability']*100:.1f}%")

        if replay_result["late_label"] == "Late":
            st.error(f"🚨 Bus **{event['trip_id']}** is predicted to be **LATE**.")
        else:
            st.success(f"✅ Bus **{event['trip_id']}** is predicted to be **ON TIME**.")

# ==============================================================================
# SECTION 4: MODEL COMPARISON
# ==============================================================================
elif section == "4. Model Comparison":
    st.header("4. Model Comparison & Metrics")
    st.write("Chronological Split: 60% Train (1,800 trips), 20% Validation (600 trips), 20% Test (600 trips).")

    st.subheader("Validation Set Comparison (Used to Select Best Regression Model):")
    if not val_metrics.empty:
        st.table(val_metrics)
    else:
        st.warning("Validation metrics not found. Run `python train_models.py`.")

    winner_name = model_meta.get("best_regression_model_name", "Linear Regression")
    winner_mae = model_meta.get("validation_mae", "2.77")
    st.info(f"🏆 Selected Model: **{winner_name}** with lowest Validation MAE = **{winner_mae} min**.")

    st.write("---")
    st.subheader("Final Test Set Evaluation (Untouched 20% Holdout):")
    if not test_metrics.empty:
        st.table(test_metrics)
    else:
        st.warning("Test metrics not found. Run `python train_models.py`.")

    st.caption("Note: Regression MAE (minutes of error) is evaluated separately from Classification Accuracy (percentage of late/on-time trips correctly categorized).")

# ==============================================================================
# SECTION 5: DATA VISUALIZATIONS
# ==============================================================================
elif section == "5. Data Visualizations":
    st.header("5. Data Visualizations")
    st.write("Visual inspection of simulated transport operations and model accuracy.")

    if df_trips.empty:
        st.error("Historical dataset empty or not found.")
    else:
        v1, v2 = st.columns(2)

        with v1:
            st.subheader("1. Late vs On-Time Trip Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            counts = df_trips['late_flag'].value_counts()
            ax1.pie(counts, labels=['On Time (<= 10 min)', 'Late (> 10 min)'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=140)
            ax1.set_title("Punctuality Distribution (N=3,000)")
            st.pyplot(fig1)

        with v2:
            st.subheader("2. Average Journey Duration by Route")
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            avg_dur = df_trips.groupby('route_id')['actual_duration_min'].mean()
            ax2.bar(avg_dur.index, avg_dur.values, color='#1976D2')
            ax2.set_ylabel("Average Duration (min)")
            ax2.set_title("Average Duration per Route")
            plt.xticks(rotation=20)
            st.pyplot(fig2)

        v3, v4 = st.columns(2)

        with v3:
            st.subheader("3. Model Validation MAE Comparison")
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            if not val_metrics.empty:
                ax3.bar(val_metrics['Model'], val_metrics['MAE_min'], color=['#90A4AE', '#2E7D32', '#FFA000', '#1565C0'])
                ax3.set_ylabel("MAE (Minutes)")
                ax3.set_title("Validation MAE (Lower is Better)")
                plt.xticks(rotation=25, ha='right')
            st.pyplot(fig3)

        with v4:
            st.subheader("4. Actual vs Scheduled Duration")
            fig4, ax4 = plt.subplots(figsize=(5, 3.5))
            sample = df_trips.sample(n=150, random_state=42)
            ax4.scatter(sample['scheduled_duration_min'], sample['actual_duration_min'], alpha=0.6, color='#5E35B1')
            ax4.plot([140, 240], [140, 240], 'r--', label='Ideal Timetable')
            ax4.set_xlabel("Scheduled Duration (min)")
            ax4.set_ylabel("Actual Duration (min)")
            ax4.set_title("Actual vs Scheduled Duration")
            ax4.legend()
            st.pyplot(fig4)

# ==============================================================================
# SECTION 6: PREDICTION HISTORY
# ==============================================================================
elif section == "6. Prediction History":
    st.header("6. Prediction History (SQLite)")
    st.write("Audit log of predictions stored in `bus_predictions.db`.")

    total_count = get_total_count()
    st.metric("Total Predictions Saved in Database", total_count)

    history = get_history(limit=50)
    if not history.empty:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No predictions logged yet. Go to Section 2 or Section 3 to generate predictions.")
