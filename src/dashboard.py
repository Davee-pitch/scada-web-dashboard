import streamlit as st
import pandas as pd
import joblib
import time
import os
import random

st.set_page_config(page_title="Industrial Control Room", layout="wide")

# ==========================================
# 1. AUTHENTICATION LAYER
# ==========================================
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

# ==========================================
# 2. MAIN DASHBOARD WITH BUILT-IN SIMULATOR
# ==========================================
st.title("⚙️ SCADA Control & Monitoring")

# Dynamically find the model path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "models", "fault_predictor.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

# Built-in generator replacing the SQLite database
def generate_live_data():
    equipment = ['MTR-01', 'PMP-01', 'FAN-01']
    data = []
    for eq in equipment:
        data.append({
            'equip_id': eq,
            'temperature_c': round(random.uniform(30.0, 95.0), 2),
            'vibration_mms': round(random.uniform(0.5, 10.0), 2),
            'current_amps': round(random.uniform(0.0, 60.0), 2),
            'operating_hours': random.randint(1000, 5000)
        })
    return pd.DataFrame(data)

df = generate_live_data()

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