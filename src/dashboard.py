import streamlit as st
import pandas as pd
import sqlite3
import joblib
import time
import os

# --- Configuration ---
st.set_page_config(page_title="Industrial Equipment Monitor", page_icon="⚙️", layout="wide")
DB_PATH = "../data/telemetry_logs.db"
MODEL_PATH = "../models/fault_predictor.pkl"

# --- Load the ML Model ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Model not found! Run ml_engine.py first.")
        st.stop()

model = load_model()

# --- Database Query ---
def get_latest_telemetry():
    """Fetches the most recent sensor reading for each piece of equipment."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT equip_id, temperature_c, vibration_mms, current_amps, operating_hours
        FROM sensor_telemetry 
        WHERE log_id IN (
            SELECT MAX(log_id) 
            FROM sensor_telemetry 
            GROUP BY equip_id
        )
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- UI Layout ---
st.title("⚙️ Smart Industrial Equipment Monitoring")
st.markdown("Real-time telemetry analysis and predictive maintenance alerts.")

# Auto-refresh toggle
col1, col2 = st.columns([8, 1])
with col2:
    auto_refresh = st.toggle("Live Data Sync", value=True)

# Fetch latest data
df_latest = get_latest_telemetry()

if df_latest.empty:
    st.warning("No telemetry data found. Please run simulator.py in the background.")
else:
    # Prepare features for the ML Model
    X_live = df_latest[['temperature_c', 'vibration_mms', 'current_amps', 'operating_hours']]
    
    # Generate predictions
    predictions = model.predict(X_live)
    df_latest['Predicted_Status'] = predictions

    # Display Metrics in Columns
    cols = st.columns(len(df_latest))
    
    for index, row in df_latest.iterrows():
        equip_id = row['equip_id']
        status = row['Predicted_Status']
        
        # Determine card color based on ML prediction
        if status == "CRITICAL":
            status_color = "🔴"
            border_color = "#ff4b4b"
        elif status == "WARNING":
            status_color = "🟡"
            border_color = "#ffa421"
        else:
            status_color = "🟢"
            border_color = "#21c354"
            
        with cols[index]:
            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 15px; background-color: #0e1117;">
                <h3 style="margin-top: 0;">{equip_id}</h3>
                <h4 style="color: {border_color}; margin-top: -10px;">{status_color} {status}</h4>
                <hr>
                <p><b>Temp:</b> {row['temperature_c']} °C</p>
                <p><b>Vibration:</b> {row['vibration_mms']} mm/s</p>
                <p><b>Current:</b> {row['current_amps']} A</p>
                <p><b>Runtime:</b> {row['operating_hours']} hrs</p>
            </div>
            """, unsafe_allow_html=True)

# Handle live refresh
if auto_refresh:
    time.sleep(2)
    st.rerun()