import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import json
import html
from datetime import datetime

st.set_page_config(page_title="Industrial Control Room", page_icon="⚙", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #eaf2f5;
        --muted: #91a5ad;
        --panel: #15252d;
        --line: #29424b;
        --mint: #55dfbd;
        --amber: #f3b64e;
        --red: #ff6a68;
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
        color: var(--ink);
        background-color: #0a151b;
        background-image: radial-gradient(#29414a 0.7px, transparent 0.7px);
        background-size: 24px 24px;
    }
    [data-testid="stHeader"] { background: rgba(10, 21, 27, 0.82); }
    [data-testid="stSidebar"] { background: #101e25; border-right: 1px solid var(--line); }
    .block-container { padding-top: 2.1rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: 0; }
    .eyebrow {
        color: var(--mint); font: 500 11px 'DM Mono', monospace;
        letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 7px;
    }
    .subtitle { color: var(--muted); font-size: 14px; margin-top: -9px; }
    [data-testid="stMetric"] {
        background: rgba(21, 37, 45, 0.92); border: 1px solid var(--line);
        border-radius: 8px; padding: 14px 17px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .equipment-card {
        background: linear-gradient(145deg, rgba(21,37,45,.98), rgba(16,30,37,.98));
        border: 1px solid var(--line); border-top: 2px solid var(--signal);
        border-radius: 8px; padding: 18px 18px 15px; margin: 2px 0 8px;
        animation: arrive .38s ease-out both;
    }
    .equipment-id { font: 500 11px 'DM Mono', monospace; color: var(--muted); }
    .equipment-state { font-size: 19px; font-weight: 800; margin: 8px 0 16px; color: var(--signal); }
    .signal-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: var(--signal); margin: 0 8px 1px 0;
        box-shadow: 0 0 12px color-mix(in srgb, var(--signal) 55%, transparent);
    }
    .critical .signal-dot { animation: signal 1.7s ease-in-out infinite; }
    .sensor-line {
        display: flex; justify-content: space-between; gap: 12px;
        border-top: 1px solid rgba(145,165,173,.13); padding-top: 9px;
        margin-top: 8px; font-size: 12px; color: var(--muted);
    }
    .sensor-line strong { color: var(--ink); font: 500 12px 'DM Mono', monospace; }
    @keyframes arrive { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes signal { 50% { opacity: .45; box-shadow: 0 0 2px var(--signal); } }
    div.stButton > button, div.stDownloadButton > button { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "models", "fault_predictor.pkl")
DB_PATH = os.path.join(project_root, "data", "telemetry_logs.db")
CONTROL_PATH = os.path.join(project_root, "data", "control.json")


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data(ttl=2)
def get_data(database_path):
    query = """
        SELECT equip_id, temperature_c, vibration_mms, current_amps, operating_hours
        FROM sensor_telemetry
        WHERE log_id IN (SELECT MAX(log_id) FROM sensor_telemetry GROUP BY equip_id)
        ORDER BY equip_id
    """
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection)


def get_controls():
    try:
        with open(CONTROL_PATH, "r", encoding="utf-8") as controls_file:
            controls = json.load(controls_file)
            return controls if isinstance(controls, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def set_control(equip_id, mode):
    controls = get_controls()
    controls[equip_id] = mode
    os.makedirs(os.path.dirname(CONTROL_PATH), exist_ok=True)
    with open(CONTROL_PATH, "w", encoding="utf-8") as controls_file:
        json.dump(controls, controls_file, indent=2)


def get_status(row):
    if row["current_amps"] < 1.0:
        return "OFFLINE", "#9aa8ae"
    status = row["Predicted_Status"]
    if status == "CRITICAL":
        return status, "#ff6a68"
    if status == "WARNING":
        return status, "#f3b64e"
    return "NORMAL", "#55dfbd"

# --- Authentication Layer ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown('<div class="eyebrow">NORTHLINE / OPERATIONS</div>', unsafe_allow_html=True)
    left, login_column, right = st.columns([1, 1.1, 1])
    with login_column:
        st.title("Control room")
        st.markdown('<p class="subtitle">Secure access to equipment health and telemetry.</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if user == "admin" and pwd == "admin123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except (OSError, ValueError, EOFError, joblib.externals.loky.process_executor.TerminatedWorkerError):
    st.error("The fault prediction model could not be loaded. Run ml_engine.py first.")
    st.stop()

st.markdown('<div class="eyebrow">NORTHLINE / OPERATIONS</div>', unsafe_allow_html=True)
st.title("Equipment intelligence")
st.markdown('<p class="subtitle">Predictive maintenance · Live telemetry</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Control panel")
    auto_refresh = st.toggle("Live updates", value=True)
    refresh_seconds = st.selectbox("Update interval", [2, 5, 10], index=1, disabled=not auto_refresh)
    status_filter = st.selectbox("Show status", ["All", "NORMAL", "WARNING", "CRITICAL", "OFFLINE"])
    if st.button("Refresh now", use_container_width=True):
        get_data.clear()
        st.rerun()
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()
    st.caption("Predictive Maintenance System")


@st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
def telemetry_panel():
    try:
        frame = get_data(DB_PATH)
    except (sqlite3.Error, OSError, pd.errors.DatabaseError) as error:
        st.error(f"Telemetry database unavailable: {error}")
        return

    if frame.empty:
        st.info("No telemetry received yet. Start the simulator to bring equipment online.")
        return

    feature_columns = ["temperature_c", "vibration_mms", "current_amps", "operating_hours"]
    frame["Predicted_Status"] = model.predict(frame[feature_columns])
    frame["Status"] = frame.apply(lambda row: get_status(row)[0], axis=1)
    counts = frame["Status"].value_counts()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Assets monitored", len(frame))
    metric_columns[1].metric("Normal", int(counts.get("NORMAL", 0)))
    metric_columns[2].metric("Warnings", int(counts.get("WARNING", 0)))
    metric_columns[3].metric("Critical / offline", int(counts.get("CRITICAL", 0) + counts.get("OFFLINE", 0)))

    visible_frame = frame if status_filter == "All" else frame[frame["Status"] == status_filter]
    st.markdown(f"#### Equipment status <span style='color:#91a5ad;font-size:13px;font-weight:400'>/ {len(visible_frame)} assets</span>", unsafe_allow_html=True)

    if visible_frame.empty:
        st.info("No equipment matches this status filter.")
    else:
        controls = get_controls()
        for batch_start in range(0, len(visible_frame), 3):
            cards = st.columns(3)
            batch = visible_frame.iloc[batch_start:batch_start + 3]
            for column, (_, row) in zip(cards, batch.iterrows()):
                status, color = get_status(row)
                equipment_id = str(row["equip_id"])
                safe_id = html.escape(equipment_id)
                card_class = "critical" if status == "CRITICAL" else ""
                with column:
                    st.markdown(
                        f"""
                        <div class="equipment-card {card_class}" style="--signal:{color}">
                            <div class="equipment-id">ASSET / {safe_id}</div>
                            <div class="equipment-state"><span class="signal-dot"></span>{status}</div>
                            <div class="sensor-line"><span>Temperature</span><strong>{row['temperature_c']:.1f} °C</strong></div>
                            <div class="sensor-line"><span>Vibration</span><strong>{row['vibration_mms']:.2f} mm/s</strong></div>
                            <div class="sensor-line"><span>Current</span><strong>{row['current_amps']:.2f} A</strong></div>
                            <div class="sensor-line"><span>Operating time</span><strong>{row['operating_hours']:.0f} h</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    current_mode = controls.get(equipment_id, "AUTO")
                    selected_mode = st.selectbox(
                        "Control mode", ["AUTO", "CRITICAL", "OFF"],
                        index=["AUTO", "CRITICAL", "OFF"].index(current_mode)
                        if current_mode in ["AUTO", "CRITICAL", "OFF"] else 0,
                        key=f"control_{equipment_id}", label_visibility="collapsed",
                    )
                    if selected_mode != current_mode:
                        set_control(equipment_id, selected_mode)
                        st.toast(f"{safe_id} set to {selected_mode}")

    st.caption(f"Last updated {datetime.now().strftime('%H:%M:%S')}")
    export_frame = frame.drop(columns=["Predicted_Status"], errors="ignore")
    st.download_button(
        "Export telemetry CSV", export_frame.to_csv(index=False),
        file_name="equipment_telemetry.csv", mime="text/csv",
    )


telemetry_panel()