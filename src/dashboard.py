import streamlit as st
import sqlite3
import pandas as pd
import joblib
import time
import os

st.set_page_config(page_title="Industrial Control Room", layout="wide")

# --- Authentication Layer ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 System Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# --- Main Dashboard ---
st.title("⚙️ SCADA Control & Monitoring")

# Dynamically find the absolute path to the main project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(project_root, "models", "fault_predictor.pkl")
MODEL_PATH = os.path.join(project_root, "models", "fault_predictor.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed looking in: {MODEL_PATH}")
    st.error(f"The exact system error is: {e}")
    st.stop()

DB_PATH = os.path.join(project_root, "data", "telemetry_logs.db")
def get_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT equip_id, temperature_c, vibration_mms, current_amps, operating_hours
        FROM sensor_telemetry 
        WHERE log_id IN (SELECT MAX(log_id) FROM sensor_telemetry GROUP BY equip_id)
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df = get_data()

if not df.empty:
    X_live = df[['temperature_c', 'vibration_mms', 'current_amps', 'operating_hours']]
    df['Predicted_Status'] = model.predict(X_live)

    cols = st.columns(len(df))
    for index, row in df.iterrows():
        with cols[index]:
            status = row['Predicted_Status']
            
            if row['current_amps'] < 1.0:
                st.info(f"**{row['equip_id']}**: ⚪ OFFLINE")
            elif status == "CRITICAL":
                st.error(f"**{row['equip_id']}**: 🔴 CRITICAL")
            elif status == "WARNING":
                st.warning(f"**{row['equip_id']}**: 🟡 WARNING")
            else:
                st.success(f"**{row['equip_id']}**: 🟢 NORMAL")
                
            st.write(f"Temp: {row['temperature_c']} °C")
            st.write(f"Vib: {row['vibration_mms']} mm/s")
            st.write(f"Curr: {row['current_amps']} A")

# Auto-refresh every 2 seconds
time.sleep(2)
st.rerun()