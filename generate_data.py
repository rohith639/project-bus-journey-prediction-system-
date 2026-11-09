"""
Synthetic Dataset Generator for Tamil Nadu Bus Journey Prediction
Author: AIML Student Internship Project (Career Solutions)

SIMULATED DATA - NOT A LIVE TRANSPORT FEED
This script creates simulated bus trip data for 5 fictional Tamil Nadu routes.
Random Seed: 42 (so results are always identical and reproducible)
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_bus_dataset():
    # Set fixed random seed for reproducible results
    np.random.seed(42)

    # Make sure data folder exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Define 5 fictional Tamil Nadu routes with standard distance and timetable time
    routes_info = {
        "TN-DEMO-01": {"name": "Chennai (CMBT) -> Vellore", "base_distance": 138.0, "base_stops": 10, "base_duration": 180},
        "TN-DEMO-02": {"name": "Coimbatore -> Salem", "base_distance": 164.0, "base_stops": 14, "base_duration": 210},
        "TN-DEMO-03": {"name": "Madurai -> Trichy", "base_distance": 132.0, "base_stops": 9, "base_duration": 170},
        "TN-DEMO-04": {"name": "Tirunelveli -> Madurai", "base_distance": 158.0, "base_stops": 12, "base_duration": 200},
        "TN-DEMO-05": {"name": "Chennai (CMBT) -> Villupuram", "base_distance": 152.0, "base_stops": 11, "base_duration": 195}
    }
    route_ids = list(routes_info.keys())

    # 2. Setup 60 simulation days (50 trips per day = 3,000 total trips)
    start_date = datetime(2026, 1, 1)
    total_days = 60
    trips_per_day = 50

    historical_records = []
    departure_events = []

    trip_number = 1

    # Loop day by day
    for day_index in range(total_days):
        current_date = start_date + timedelta(days=day_index)
        date_str = current_date.strftime("%Y-%m-%d")
        weekday_name = current_date.strftime("%A")
        is_weekend = weekday_name in ["Saturday", "Sunday"]

        # Loop trip by trip for each day
        for _ in range(trips_per_day):
            trip_id = f"TRIP_{trip_number:04d}"
            trip_number += 1

            # Pick a route at random
            route_id = np.random.choice(route_ids)
            route = routes_info[route_id]

            # Realistic distance (+/- 1.5 km variation due to minor diversions)
            route_distance_km = round(route["base_distance"] + np.random.uniform(-1.5, 1.5), 1)

            # Scheduled duration in timetable (+/- 5 minutes timetable slot variation)
            scheduled_duration_min = int(route["base_duration"] + np.random.choice([-5, 0, 5]))

            # Stop count (base stops +/- 2)
            stop_count = int(np.clip(route["base_stops"] + np.random.randint(-2, 3), 5, 25))

            # Departure hour between 5 AM and 10 PM
            departure_hour = int(np.random.choice(range(5, 23)))
            departure_minute = int(np.random.choice([0, 10, 15, 20, 30, 40, 45, 50]))
            departure_timestamp = f"{date_str} {departure_hour:02d}:{departure_minute:02d}:00"

            # Traffic index between 0.05 and 0.98
            # Higher traffic during rush hours (8-10 AM and 5-8 PM) and on weekends
            traffic = np.random.uniform(0.15, 0.45)
            if departure_hour in [8, 9, 10]:
                traffic += np.random.uniform(0.20, 0.35)
            elif departure_hour in [17, 18, 19, 20]:
                traffic += np.random.uniform(0.25, 0.40)
            
            if is_weekend:
                traffic += np.random.uniform(0.05, 0.15)
            
            departure_traffic_index = round(float(np.clip(traffic, 0.05, 0.98)), 2)

            # Rainfall in mm (80% chance dry, 20% chance rain up to 35 mm)
            if np.random.rand() > 0.80:
                rain_mm_last_hour = round(float(np.random.exponential(scale=6.0) + 1.0), 1)
                rain_mm_last_hour = min(rain_mm_last_hour, 35.0)
            else:
                rain_mm_last_hour = 0.0

            # 3. Calculate actual journey duration using a simple physical formula:
            # - baseline timetable duration
            # - extra traffic delay (above normal traffic level 0.28)
            # - rain delay (up to 16 extra minutes in heavy rain)
            # - extra stops delay (1.8 minutes per extra stop)
            # - random noise (+/- 3.5 minutes for driver variation)
            traffic_delay = (departure_traffic_index - 0.28) * 32.0
            rain_delay = (rain_mm_last_hour / 35.0) * 16.0
            stop_delay = (stop_count - route["base_stops"]) * 1.8
            random_noise = np.random.normal(loc=0.0, scale=3.5)

            actual_duration_min = scheduled_duration_min + traffic_delay + rain_delay + stop_delay + random_noise

            # Safety check: bus duration is never negative or unreasonably low (at least 80% of schedule)
            actual_duration_min = round(max(scheduled_duration_min * 0.80, actual_duration_min), 1)

            # 4. Late Flag Rule:
            # Bus is late (1) if it takes more than 10 minutes past scheduled time; otherwise on time (0)
            if actual_duration_min > (scheduled_duration_min + 10.0):
                late_flag = 1
            else:
                late_flag = 0

            # Historical record: contains everything including actual outcomes for training
            historical_record = {
                "trip_id": trip_id,
                "route_id": route_id,
                "departure_date": date_str,
                "route_distance_km": route_distance_km,
                "scheduled_duration_min": scheduled_duration_min,
                "departure_hour": departure_hour,
                "weekday": weekday_name,
                "stop_count": stop_count,
                "departure_traffic_index": departure_traffic_index,
                "rain_mm_last_hour": rain_mm_last_hour,
                "actual_duration_min": actual_duration_min,
                "late_flag": late_flag
            }
            historical_records.append(historical_record)

            # Departure event: ONLY information available at departure time (NO LEAKAGE)
            departure_event = {
                "trip_id": trip_id,
                "route_id": route_id,
                "departure_date": date_str,
                "departure_timestamp": departure_timestamp,
                "departure_hour": departure_hour,
                "weekday": weekday_name,
                "route_distance_km": route_distance_km,
                "scheduled_duration_min": scheduled_duration_min,
                "stop_count": stop_count,
                "departure_traffic_index": departure_traffic_index,
                "rain_mm_last_hour": rain_mm_last_hour
            }
            departure_events.append(departure_event)

    # Convert to DataFrames
    df_trips = pd.DataFrame(historical_records)
    df_events = pd.DataFrame(departure_events)

    # Save to CSV files
    trips_path = os.path.join(data_dir, "trips.csv")
    events_path = os.path.join(data_dir, "departure_events.csv")

    df_trips.to_csv(trips_path, index=False)
    df_events.to_csv(events_path, index=False)

    print("==========================================")
    print("DATASET GENERATION FINISHED SUCCESSFULLY")
    print("==========================================")
    print(f"Total trips generated: {len(df_trips)}")
    print(f"Saved historical trips: {trips_path}")
    print(f"Saved departure events: {events_path}")
    print(f"On-time trips: {(df_trips['late_flag'] == 0).sum()} ({(1 - df_trips['late_flag'].mean())*100:.1f}%)")
    print(f"Late trips:    {df_trips['late_flag'].sum()} ({df_trips['late_flag'].mean()*100:.1f}%)")

if __name__ == "__main__":
    generate_bus_dataset()
