import sqlite3
import os

DB_PATH = "../data/telemetry_logs.db"

def setup_database():
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: Equipment Registry (The physical inventory)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS equipment_registry (
        equip_id TEXT PRIMARY KEY,
        equip_type TEXT,
        baseline_temp REAL,
        baseline_vibration REAL,
        baseline_current REAL
    )
    ''')

    # Table 2: Sensor Telemetry (The real-time time-series data)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_telemetry (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        equip_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        temperature_c REAL,
        vibration_mms REAL,
        current_amps REAL,
        operating_hours INTEGER,
        status_label TEXT,
        FOREIGN KEY(equip_id) REFERENCES equipment_registry(equip_id)
    )
    ''')

    conn.commit()
    conn.close()
    print("Database schema initialized successfully.")

def seed_equipment():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Baseline specs: [ID, Type, Temp(C), Vibration(mm/s), Current(A)]
    equipment_list = [
        ('MTR-01', '3-Phase Induction Motor', 65.0, 2.5, 30.0),
        ('PMP-01', 'Centrifugal Pump', 55.0, 1.8, 45.0),
        ('GEN-01', 'Backup Generator', 85.0, 4.0, 120.0)
    ]
    
    cursor.executemany('''
    INSERT OR IGNORE INTO equipment_registry 
    (equip_id, equip_type, baseline_temp, baseline_vibration, baseline_current)
    VALUES (?, ?, ?, ?, ?)
    ''', equipment_list)
    
    conn.commit()
    conn.close()
    print("Equipment registry seeded.")

if __name__ == "__main__":
    setup_database()
    seed_equipment()