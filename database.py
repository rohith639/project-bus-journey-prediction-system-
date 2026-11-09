"""
SQLite Database Management for Bus Journey Prediction System
Author: AIML Student Internship Project (Career Solutions)

SIMULATED DATA - NOT A LIVE TRANSPORT FEED
"""

import os
import sqlite3
import pandas as pd

# Default SQLite database file path
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bus_predictions.db"
)

def init_db():
    """Initializes SQLite database and creates the two required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: trip_events (stores departure events)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trip_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT NOT NULL,
            route_id TEXT NOT NULL,
            departure_timestamp TEXT NOT NULL,
            input_data TEXT,
            source TEXT DEFAULT 'Manual Input',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Table 2: predictions (stores model prediction outputs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT NOT NULL,
            predicted_duration_min REAL NOT NULL,
            predicted_arrival TEXT NOT NULL,
            late_probability REAL NOT NULL,
            late_label TEXT NOT NULL,
            model_version TEXT NOT NULL,
            predicted_at TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()

def log_trip_event_and_prediction(
    trip_id: str,
    route_id: str,
    departure_timestamp: str,
    input_data_json: str,
    source: str,
    predicted_duration_min: float,
    predicted_arrival: str,
    late_probability: float,
    late_label: str,
    model_version: str,
    predicted_at: str
):
    """Saves a trip departure event and its model prediction in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Save to trip_events table
    cursor.execute("""
        INSERT INTO trip_events (trip_id, route_id, departure_timestamp, input_data, source)
        VALUES (?, ?, ?, ?, ?)
    """, (trip_id, route_id, departure_timestamp, input_data_json, source))

    # Save to predictions table
    cursor.execute("""
        INSERT INTO predictions (trip_id, predicted_duration_min, predicted_arrival, late_probability, late_label, model_version, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (trip_id, predicted_duration_min, predicted_arrival, late_probability, late_label, model_version, predicted_at))

    conn.commit()
    conn.close()

def get_prediction_history(limit: int = 50) -> pd.DataFrame:
    """Fetches the latest prediction history from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            p.id,
            p.trip_id,
            t.route_id,
            t.departure_timestamp,
            p.predicted_duration_min,
            p.predicted_arrival,
            ROUND(p.late_probability * 100, 1) || '%' AS late_probability_pct,
            p.late_label,
            p.model_version,
            p.predicted_at,
            t.source
        FROM predictions p
        LEFT JOIN trip_events t ON p.trip_id = t.trip_id
        ORDER BY p.id DESC
        LIMIT {limit};
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_prediction_count() -> int:
    """Returns the total number of saved predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        count = cursor.fetchone()[0]
    except Exception:
        count = 0
    finally:
        conn.close()
    return count

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
