import json
import time
import sqlite3
import random
from datetime import datetime

DB_PATH = "../data/telemetry_logs.db"

def get_equipment():
    """Fetch baseline parameters from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT equip_id, baseline_temp, baseline_vibration, baseline_current FROM equipment_registry")
    equipment = cursor.fetchall()
    conn.close()
    return equipment

def determine_status(temp_diff, vib_diff, current_diff):
    """
    Physical thresholds to label the data. 
    The ML model will later learn to predict these labels without explicit rules.
    """
    if temp_diff > 20 or vib_diff > 3.0 or current_diff > 15:
        return "CRITICAL"
    elif temp_diff > 10 or vib_diff > 1.5 or current_diff > 7:
        return "WARNING"
    return "NORMAL"

def read_controls():
    """Reads commands from the frontend UI."""
    try:
        with open("../data/control.json", "r") as f:
            return json.load(f)
    except:
        return {} # Default if file doesn't exist yet

def run_simulation(interval_seconds=2, logs_to_generate=1000):
    equipment_data = get_equipment()
    
    # Track operating hours
    operating_hours = {row[0]: random.randint(2000, 8000) for row in equipment_data}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Starting industrial equipment simulation with live UI controls...")
    
    for i in range(logs_to_generate):
        controls = read_controls()
        
        for equip in equipment_data:
            equip_id, base_temp, base_vib, base_curr = equip
            
            # Check what the UI is commanding this specific machine to do
            command = controls.get(equip_id, "AUTO")
            
            temp = random.gauss(base_temp, 2.0)
            vib = random.gauss(base_vib, 0.2)
            curr = random.gauss(base_curr, 1.0)
            
            if command == "OFF":
                temp, vib, curr = 32.0, 0.0, 0.0  # Powered down
            elif command == "CRITICAL":
                temp += random.uniform(50, 80)    # Forced overload
                vib += random.uniform(8.0, 15.0)
                curr += random.uniform(20, 40)
            else: # "AUTO" mode
                if random.random() < 0.10:        # 10% chance of random anomaly
                    temp += random.uniform(10, 25)
                    vib += random.uniform(1.0, 4.0)
                    curr += random.uniform(5, 20)

            # Calculate deviations for the label
            temp_diff = temp - base_temp
            vib_diff = vib - base_vib
            current_diff = curr - base_curr
            
            status_label = determine_status(temp_diff, vib_diff, current_diff)
            if command == "OFF": 
                status_label = "NORMAL"

            # Insert into database
            cursor.execute('''
                INSERT INTO sensor_telemetry 
                (equip_id, temperature_c, vibration_mms, current_amps, operating_hours, status_label)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (equip_id, round(temp, 2), round(vib, 2), round(curr, 2), round(operating_hours[equip_id], 2), status_label))
            
        conn.commit()
        print(f"Log {i+1} inserted (Reading commands from UI...)")
        time.sleep(interval_seconds)
        
    conn.close()

if __name__ == "__main__":
    run_simulation()