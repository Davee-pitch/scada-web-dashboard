import customtkinter as ctk
import sqlite3
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import json
import sys #
import tkinter as tk
from tkinter import messagebox

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Determine exactly where the .exe or .py file is located
if getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(sys.executable)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Step one level up to the main project folder
project_root = os.path.dirname(current_dir)

# Build absolute paths that Windows cannot misinterpret
DB_PATH = os.path.join(project_root, "data", "telemetry_logs.db")
MODEL_PATH = os.path.join(project_root, "models", "fault_predictor.pkl")
CONTROL_PATH = os.path.join(project_root, "data", "control.json")
class PredictiveMaintenanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Industrial Control Room")
        self.geometry("1100x650")
        self.minsize(900, 560)
        self.refresh_job = None
        self.animation_job = None
        self.auto_refresh = True
        self.is_logged_in = False
        self.protocol("WM_DELETE_WINDOW", self.confirm_close)
        
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None

        # Start with the login screen instead of the dashboard
        self.build_login_screen()

    def build_login_screen(self):
        self.configure(fg_color="#0b1220")

        self.login_frame = ctk.CTkFrame(
            self,
            width=390,
            height=430,
            corner_radius=20,
            fg_color="#111c2e",
            border_width=1,
            border_color="#263b5a"
        )
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.login_frame.pack_propagate(False)

        ctk.CTkLabel(
            self.login_frame,
            text="⚙",
            font=("Arial", 44),
            text_color="#3b9eff"
        ).pack(pady=(28, 0))

        ctk.CTkLabel(
            self.login_frame,
            text="INDUSTRIAL CONTROL ROOM",
            font=("Arial", 20, "bold"),
            text_color="#ffffff"
        ).pack(pady=(4, 2))

        ctk.CTkLabel(
            self.login_frame,
            text="Predictive Maintenance System",
            font=("Arial", 13),
            text_color="#8fa8c7"
        ).pack(pady=(0, 24))

        self.username_entry = ctk.CTkEntry(
            self.login_frame,
            placeholder_text="Username",
            width=285,
            height=42,
            corner_radius=10
        )
        self.username_entry.pack(pady=8)

        self.password_entry = ctk.CTkEntry(
            self.login_frame,
            placeholder_text="Password",
            show="*",
            width=285,
            height=42,
            corner_radius=10
        )
        self.password_entry.pack(pady=8)

        self.error_label = ctk.CTkLabel(
            self.login_frame,
            text="",
            text_color="#ff6262",
            font=("Arial", 12)
        )
        self.error_label.pack(pady=(4, 2))

        ctk.CTkButton(
            self.login_frame,
            text="SIGN IN",
            width=285,
            height=42,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#2478d4",
            hover_color="#1d62ad",
            command=self.verify_login
        ).pack(pady=(12, 10))

        ctk.CTkLabel(
            self.login_frame,
            text="Authorized personnel only",
            font=("Arial", 11),
            text_color="#7186a3"
        ).pack(pady=(4, 0))

        self.password_entry.bind("<Return>", lambda event: self.verify_login())

    def verify_login(self):
        # Default prototype credentials
        if self.username_entry.get() == "admin" and self.password_entry.get() == "admin123":
            self.login_frame.destroy()
            self.is_logged_in = True
            self.build_dashboard()
        else:
            self.error_label.configure(text="Invalid credentials")
            self.password_entry.delete(0, "end")

    def logout(self):
        self.is_logged_in = False
        self.cancel_refresh()
        self.cancel_animation()
        self.dashboard_frame.destroy()
        self.build_login_screen()

    def confirm_close(self):
        if messagebox.askyesno("Exit", "Close the Industrial Control Room?"):
            self.is_logged_in = False
            self.cancel_refresh()
            self.cancel_animation()
            self.destroy()

    def build_dashboard(self):
        self.dashboard_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.dashboard_frame.pack(fill="both", expand=True, padx=24, pady=20)

        toolbar = ctk.CTkFrame(self.dashboard_frame, fg_color="#111c2e", corner_radius=12)
        toolbar.pack(fill="x", pady=(0, 18))

        self.circuit_canvas = tk.Canvas(
            toolbar, width=190, height=82, bg="#111c2e", highlightthickness=0
        )
        self.circuit_canvas.pack(side="left", padx=(12, 4), pady=8)
        self.draw_circuit_animation()

        title_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        title_frame.pack(side="left", fill="y", expand=True, padx=12, pady=15)
        ctk.CTkLabel(
            title_frame, text="SCADA CONTROL ROOM", font=("Arial", 21, "bold"),
            text_color="#ffffff", anchor="w"
        ).pack(anchor="w")
        self.last_updated_label = ctk.CTkLabel(
            title_frame, text="Live telemetry", font=("Arial", 12),
            text_color="#8fa8c7", anchor="w"
        )
        self.last_updated_label.pack(anchor="w", pady=(4, 0))

        actions = ctk.CTkFrame(toolbar, fg_color="transparent")
        actions.pack(side="right", padx=14, pady=14)
        self.auto_refresh_switch = ctk.CTkSwitch(
            actions, text="Live sync", command=self.toggle_auto_refresh,
            onvalue=True, offvalue=False
        )
        self.auto_refresh_switch.select()
        self.auto_refresh_switch.pack(side="left", padx=(0, 14))
        ctk.CTkButton(
            actions, text="Refresh", width=86, command=self.refresh_now
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            actions, text="About", width=72, fg_color="#263b5a",
            hover_color="#304b70", command=self.show_about
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            actions, text="Log out", width=82, fg_color="#8f3540",
            hover_color="#702a33", command=self.logout
        ).pack(side="left", padx=4)

        self.cards_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True)
        self.update_dashboard()

    def show_about(self):
        messagebox.showinfo(
            "About Industrial Control Room",
            "Predictive Maintenance System\nSCADA telemetry and equipment status monitoring"
        )

    def toggle_auto_refresh(self):
        self.auto_refresh = bool(self.auto_refresh_switch.get())
        self.cancel_refresh()
        if self.auto_refresh:
            self.schedule_refresh()

    def refresh_now(self):
        self.cancel_refresh()
        self.update_dashboard()

    def schedule_refresh(self):
        if self.is_logged_in and self.auto_refresh and self.refresh_job is None:
            self.refresh_job = self.after(2000, self.update_dashboard)

    def cancel_refresh(self):
        if self.refresh_job is not None:
            self.after_cancel(self.refresh_job)
            self.refresh_job = None

    def cancel_animation(self):
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None

    def draw_circuit_animation(self):
        canvas = self.circuit_canvas
        routes = [
            [(8, 20), (42, 20), (42, 10), (86, 10), (86, 34), (126, 34), (126, 20), (178, 20)],
            [(8, 42), (28, 42), (28, 64), (72, 64), (72, 48), (112, 48), (112, 70), (178, 70)],
            [(8, 62), (18, 62), (18, 32), (58, 32), (58, 48)],
        ]
        self.circuit_routes = routes
        self.circuit_pulses = []
        for route in routes:
            flattened = " ".join(f"{x},{y}" for x, y in route)
            canvas.create_line(
                *[coordinate for point in route for coordinate in point],
                fill="#254667", width=2, smooth=True
            )
            for x, y in route:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#3c6b8d", outline="")
            start_x, start_y = route[0]
            pulse_id = canvas.create_oval(
                start_x - 3, start_y - 3, start_x + 3, start_y + 3,
                fill="#4bd8c7", outline=""
            )
            self.circuit_pulses.append((pulse_id, 0))
        self.animate_circuit()

    def animate_circuit(self):
        if not self.is_logged_in:
            return
        updated_pulses = []
        for route_index, (pulse_id, point_index) in enumerate(self.circuit_pulses):
            route = self.circuit_routes[route_index]
            point_index = (point_index + 1) % len(route)
            x, y = route[point_index]
            self.circuit_canvas.coords(pulse_id, x - 3, y - 3, x + 3, y + 3)
            updated_pulses.append((pulse_id, point_index))
        self.circuit_pulses = updated_pulses
        self.animation_job = self.after(180, self.animate_circuit)

    def get_controls(self):
        try:
            with open(CONTROL_PATH, "r") as f:
                return json.load(f)
        except:
            return {"MTR-01": "AUTO", "PMP-01": "AUTO", "GEN-01": "AUTO"}

    def set_control(self, equip_id, mode):
        controls = self.get_controls()
        controls[equip_id] = mode
        os.makedirs(os.path.dirname(CONTROL_PATH), exist_ok=True)
        with open(CONTROL_PATH, "w") as f:
            json.dump(controls, f)

    def get_latest_telemetry(self):
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT equip_id, temperature_c, vibration_mms, current_amps, operating_hours
            FROM sensor_telemetry 
            WHERE log_id IN (SELECT MAX(log_id) FROM sensor_telemetry GROUP BY equip_id)
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def update_dashboard(self):
        self.refresh_job = None
        if not self.is_logged_in:
            return

        try:
            df = self.get_latest_telemetry()
            controls = self.get_controls()
        except (sqlite3.Error, OSError, pd.errors.DatabaseError) as error:
            df = pd.DataFrame()
            self.last_updated_label.configure(text=f"Unable to load telemetry: {error}")

        for widget in self.cards_frame.winfo_children():
            widget.destroy()

        if df.empty:
            message = "No telemetry data available. Start the simulator to receive live readings."
            if self.model is None:
                message = "Model not found. Run ml_engine.py before monitoring equipment."
            ctk.CTkLabel(
                self.cards_frame, text=message, font=("Arial", 15),
                text_color="#8fa8c7", wraplength=720
            ).pack(pady=60)
        elif self.model:
            X_live = df[['temperature_c', 'vibration_mms', 'current_amps', 'operating_hours']]
            df['Predicted_Status'] = self.model.predict(X_live)

            for index, row in df.iterrows():
                equip_id = row['equip_id']
                status = row['Predicted_Status']
                current_mode = controls.get(equip_id, "AUTO")
                
                if row['current_amps'] < 1.0:
                    border_color = "#555555"
                    status_text = "⚪ OFFLINE"
                elif status == "CRITICAL":
                    border_color = "#ff4b4b"
                    status_text = "🔴 CRITICAL"
                elif status == "WARNING":
                    border_color = "#ffa421"
                    status_text = "🟡 WARNING"
                else:
                    border_color = "#21c354"
                    status_text = "🟢 NORMAL"

                card = ctk.CTkFrame(self.cards_frame, border_width=2, border_color=border_color, corner_radius=10)
                card.pack(side="left", fill="both", expand=True, padx=10)

                ctk.CTkLabel(card, text=equip_id, font=("Arial", 20, "bold")).pack(pady=(15, 5))
                ctk.CTkLabel(card, text=status_text, font=("Arial", 16, "bold"), text_color=border_color).pack(pady=(0, 10))
                
                control_btn = ctk.CTkSegmentedButton(
                    card, 
                    values=["AUTO", "CRITICAL", "OFF"],
                    command=lambda mode, e=equip_id: self.set_control(e, mode)
                )
                control_btn.set(current_mode)
                control_btn.pack(pady=5)

                ctk.CTkLabel(card, text=f"Temp: {row['temperature_c']} °C", font=("Arial", 14)).pack(pady=(10,0))
                ctk.CTkLabel(card, text=f"Vib: {row['vibration_mms']} mm/s", font=("Arial", 14)).pack()
                ctk.CTkLabel(card, text=f"Curr: {row['current_amps']} A", font=("Arial", 14)).pack(pady=(0, 15))

            if df.shape[0] and self.model:
                self.last_updated_label.configure(text="Telemetry updated just now")
            self.schedule_refresh()

if __name__ == "__main__":
    app = PredictiveMaintenanceApp()
    app.mainloop()