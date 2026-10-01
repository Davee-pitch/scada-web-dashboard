# ⚙️ Industrial SCADA Predictive Maintenance Dashboard

## Overview
A web-based SCADA monitoring system that ingests live, simulated industrial hardware telemetry (temperature, vibration, current) and utilizes machine learning to predict equipment failure before it happens.

**[🔴 Live Mobile Demo Here] (https://scada-web-dashboard-eslvket3mj92cmtfrf8od3.streamlit.app/)**
* **Username:** admin
* **Password:** admin123

## Technical Architecture
* **Frontend/Deployment:** Streamlit Community Cloud
* **Machine Learning:** Scikit-learn (Random Forest/Fault Prediction)
* **Data Processing:** Pandas, NumPy
* **Language:** Python

## Core Features
* **Real-Time Telemetry Simulation:** Generates live sensor fluctuations every 2 seconds.
* **Predictive Diagnostics:** ML model automatically categorizes machine states (NORMAL, WARNING, CRITICAL, OFFLINE).
* **Secure Access:** Built-in session state authentication layer.
