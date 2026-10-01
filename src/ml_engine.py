import sqlite3
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DB_PATH = "../data/telemetry_logs.db"
MODEL_DIR = "../models"
MODEL_PATH = f"{MODEL_DIR}/fault_predictor.pkl"

def load_data():
    """Extracts the time-series data from SQLite into a Pandas DataFrame."""
    print("Connecting to database and extracting telemetry logs...")
    conn = sqlite3.connect(DB_PATH)
    
    # We join with the registry to get the baseline features if we want them, 
    # but for now, the raw readings + operating hours are our primary features.
    query = """
        SELECT temperature_c, vibration_mms, current_amps, operating_hours, status_label 
        FROM sensor_telemetry
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    if len(df) < 50:
        print("Warning: Not much data yet. Run simulator.py longer for better ML results.")
        
    return df

def train_model():
    df = load_data()
    
    # 1. Prepare Features (X) and Target (y)
    X = df[['temperature_c', 'vibration_mms', 'current_amps', 'operating_hours']]
    y = df['status_label']
    
    # 2. Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Data: {len(X_train)} samples")
    print(f"Testing Data: {len(X_test)} samples")
    
    # 3. Initialize and Train the Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    # This report shows Precision and Recall - crucial metrics for CV projects
    print(classification_report(y_test, y_pred))
    
    # 5. Save the trained model to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()