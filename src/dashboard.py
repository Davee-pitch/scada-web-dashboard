import streamlit as st
import pandas as pd
import joblib
import os
import random
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
        box-shadow: 0 0 12px var(--signal);
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fault_predictor.pkl")
EQUIPMENT = {
    "MTR-01": {"temperature_c": 56.0, "vibration_mms": 2.2, "current_amps": 24.0, "operating_hours": 2840},
    "PMP-01": {"temperature_c": 48.0, "vibration_mms": 1.8, "current_amps": 17.0, "operating_hours": 1930},
    "FAN-01": {"temperature_c": 42.0, "vibration_mms": 1.3, "current_amps": 11.0, "operating_hours": 1260},
}
FEATURE_COLUMNS = ["temperature_c", "vibration_mms", "current_amps", "operating_hours"]


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def initialize_simulation():
    if "telemetry" not in st.session_state:
        st.session_state.telemetry = {
            equipment_id: {
                **readings,
                "temperature_c": round(readings["temperature_c"] + random.uniform(-2, 2), 2),
                "vibration_mms": round(readings["vibration_mms"] + random.uniform(-0.2, 0.2), 2),
            }
            for equipment_id, readings in EQUIPMENT.items()
        }
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = []
    if "fault_asset" not in st.session_state:
        st.session_state.fault_asset = None


def advance_simulation():
    initialize_simulation()
    for equipment_id, readings in st.session_state.telemetry.items():
        if st.session_state.fault_asset == equipment_id:
            readings["temperature_c"] = min(98, readings["temperature_c"] + random.uniform(0.2, 1.0))
            readings["vibration_mms"] = min(12, readings["vibration_mms"] + random.uniform(0.1, 0.35))
            readings["current_amps"] = min(60, readings["current_amps"] + random.uniform(0.2, 1.0))
        else:
            readings["temperature_c"] = min(90, max(25, readings["temperature_c"] + random.uniform(-1.2, 1.2)))
            readings["vibration_mms"] = min(9.5, max(0.3, readings["vibration_mms"] + random.uniform(-0.18, 0.18)))
            readings["current_amps"] = min(58, max(0, readings["current_amps"] + random.uniform(-1.5, 1.5)))

    timestamp = datetime.now()
    for equipment_id, readings in st.session_state.telemetry.items():
        st.session_state.telemetry_history.append({
            "timestamp": timestamp,
            "equip_id": equipment_id,
            **readings,
        })
    st.session_state.telemetry_history = st.session_state.telemetry_history[-180:]
    return pd.DataFrame.from_dict(st.session_state.telemetry, orient="index").rename_axis("equip_id").reset_index()


def equipment_status(row):
    if row["current_amps"] < 1.0:
        return "OFFLINE", "#9aa8ae"
    status = row["Predicted_Status"]
    if status == "CRITICAL":
        return status, "#ff6a68"
    if status == "WARNING":
        return status, "#f3b64e"
    return "NORMAL", "#55dfbd"

# ==========================================
# 1. AUTHENTICATION LAYER
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown('<div class="eyebrow">NORTHLINE / OPERATIONS</div>', unsafe_allow_html=True)
    _, login_column, _ = st.columns([1, 1.1, 1])
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

# ==========================================
# 2. MAIN DASHBOARD WITH BUILT-IN SIMULATOR
# ==========================================
st.markdown('<div class="eyebrow">NORTHLINE / OPERATIONS</div>', unsafe_allow_html=True)
st.title("Equipment intelligence")
st.markdown('<p class="subtitle">Predictive maintenance · Simulated live telemetry</p>', unsafe_allow_html=True)

try:
    model = load_model(MODEL_PATH)
except Exception:
    st.error("The prediction model could not be loaded. Run ml_engine.py first.")
    st.stop()

initialize_simulation()
with st.sidebar:
    st.markdown("### Control panel")
    auto_refresh = st.toggle("Live updates", value=True)
    refresh_seconds = st.selectbox("Update interval", [2, 5, 10], index=1, disabled=not auto_refresh)
    status_filter = st.selectbox("Show status", ["All", "NORMAL", "WARNING", "CRITICAL", "OFFLINE"])
    fault_asset = st.selectbox("Fault simulation", ["None", *EQUIPMENT.keys()])
    st.session_state.fault_asset = None if fault_asset == "None" else fault_asset
    if st.button("Refresh now", use_container_width=True):
        st.rerun()
    if st.button("Reset simulation", use_container_width=True):
        st.session_state.pop("telemetry", None)
        st.session_state.pop("telemetry_history", None)
        st.session_state.fault_asset = None
        st.rerun()
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()
    st.caption("Fault simulation raises the selected asset's sensor readings.")


@st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
def telemetry_panel():
    frame = advance_simulation()
    frame["Predicted_Status"] = model.predict(frame[FEATURE_COLUMNS])
    statuses = frame.apply(lambda row: equipment_status(row)[0], axis=1)
    frame["Status"] = statuses
    counts = statuses.value_counts()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Assets monitored", len(frame))
    metric_columns[1].metric("Normal", int(counts.get("NORMAL", 0)))
    metric_columns[2].metric("Warnings", int(counts.get("WARNING", 0)))
    metric_columns[3].metric("Critical / offline", int(counts.get("CRITICAL", 0) + counts.get("OFFLINE", 0)))

    visible_frame = frame if status_filter == "All" else frame[frame["Status"] == status_filter]
    st.markdown(
        f"#### Equipment status <span style='color:#91a5ad;font-size:13px;font-weight:400'>/ {len(visible_frame)} assets</span>",
        unsafe_allow_html=True,
    )
    if visible_frame.empty:
        st.info("No equipment matches this status filter.")
    else:
        for batch_start in range(0, len(visible_frame), 3):
            columns = st.columns(3)
            batch = visible_frame.iloc[batch_start:batch_start + 3]
            for column, (_, row) in zip(columns, batch.iterrows()):
                status, color = equipment_status(row)
                safe_id = html.escape(str(row["equip_id"]))
                critical_class = "critical" if status == "CRITICAL" else ""
                with column:
                    st.markdown(
                        f"""
                        <div class="equipment-card {critical_class}" style="--signal:{color}">
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

    history = pd.DataFrame(st.session_state.telemetry_history)
    with st.expander("Telemetry trends", expanded=False):
        if history.empty:
            st.caption("Trend history appears after the next update.")
        else:
            asset = st.selectbox("Equipment", list(EQUIPMENT), key="trend_asset")
            measure = st.selectbox(
                "Sensor", ["temperature_c", "vibration_mms", "current_amps"],
                format_func=lambda value: {
                    "temperature_c": "Temperature (°C)",
                    "vibration_mms": "Vibration (mm/s)",
                    "current_amps": "Current (A)",
                }[value],
                key="trend_measure",
            )
            trend = history[history["equip_id"] == asset].set_index("timestamp")[[measure]]
            st.line_chart(trend, height=220)

    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')} · Simulated sensor feed")
    st.download_button(
        "Export telemetry CSV",
        frame.drop(columns=["Predicted_Status"], errors="ignore").to_csv(index=False),
        file_name="equipment_telemetry.csv",
        mime="text/csv",
    )


telemetry_panel()