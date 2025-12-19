import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image, display



class ManufacturingDSS:
    def __init__(self, csv_path, window_size=10, k=7):
        
        self.df = pd.read_csv(csv_path)

        self.features = [
            "temperature_c",
            "pressure_bar",
            "vibration_mm_s",
            "speed_rpm",
        ]

        
        self.min_train = self.df[self.features].min()
        self.max_train = self.df[self.features].max()

      -
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[self.features])

        self.y_defect = self.df["defect_type"]
        self.y_action = self.df["recommended_action"]

        self.knn_defect = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.knn_defect.fit(self.X, self.y_defect)

        self.knn_action = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.knn_action.fit(self.X, self.y_action)

        
        golden = self.df[self.df["defect_type"] == "No Defect"]
        if len(golden) < 5:
            golden = self.df

        self.golden_center = golden[self.features].mean().values
        self.golden_std = golden[self.features].std().replace(0, 1e-6).values

        
        self.history = deque(maxlen=window_size)
        self.last_mode = "MANUAL_ONLY"
        self.consecutive_auto = 0

        self.units_per_rev = 0.05  
    def golden_score(self, raw):
        raw = np.array(raw, dtype=float)
        z = (raw - self.golden_center) / self.golden_std
        feature_scores = np.maximum(0, 1 - (np.abs(z) / 6))
        score = float(np.mean(feature_scores))
        return score, z

  
    def _ood_check(self, raw):
        return (raw < self.min_train.values).any() or (raw > self.max_train.values).any()

    
    def analyze(self, temp, pressure, vibration, speed):
        raw = np.array([temp, pressure, vibration, speed], dtype=float)

        # -------- Layer 0: Emergency Interlocks
        if vibration > 12.0 or speed < 50:
            return self._report(
                "EMERGENCY_STOP",
                "Safety violation",
                0.0,
                0.0,
                raw,
                None,
            )

        # -------- Layer 1: OOD Check
        if self._ood_check(raw):
            return self._report(
                "MANUAL_ONLY",
                "Input out of training range",
                0.0,
                0.0,
                raw,
                None,
            )

        scaled = self.scaler.transform(raw.reshape(1, -1))
        defect = self.knn_defect.predict(scaled)[0]
        action = self.knn_action.predict(scaled)[0]
        confidence = float(np.max(self.knn_defect.predict_proba(scaled)))

        golden, z = self.golden_score(raw)

        AUTO_CONF, AUTO_G = 0.80, 0.85
        HIL_CONF, HIL_G = 0.60, 0.70

        if confidence >= AUTO_CONF and golden >= AUTO_G:
            candidate = "AUTO (SCADA)"
        elif confidence >= HIL_CONF and golden >= HIL_G:
            candidate = "HUMAN_IN_LOOP (MES)"
        else:
            candidate = "MANUAL_ONLY"

        if candidate.startswith("AUTO"):
            self.consecutive_auto += 1
        else:
            self.consecutive_auto = 0

        if self.last_mode.startswith("AUTO"):
            mode = candidate if not candidate.startswith("AUTO") else "AUTO (SCADA)"
        else:
            mode = "AUTO (SCADA)" if self.consecutive_auto >= 3 else candidate

        self.last_mode = mode

        theoretical_uph = speed * self.units_per_rev * 60
        mode_factor = 1.0 if mode.startswith("AUTO") else 0.75 if mode.startswith("HUMAN") else 0.5
        predicted_uph = theoretical_uph * confidence * mode_factor

        return {
            "mode": mode,
            "defect": defect,
            "confidence": round(confidence, 2),
            "golden_score": round(golden, 2),
            "action": action if mode != "MANUAL_ONLY" else "Manual Inspection",
            "theoretical_uph": round(theoretical_uph, 1),
            "predicted_uph": round(predicted_uph, 1),
            "z_scores": np.round(z, 2),
        }

    def _report(self, mode, reason, conf, golden, raw, z):
        return {
            "mode": mode,
            "defect": "N/A",
            "confidence": conf,
            "golden_score": golden,
            "action": reason,
            "theoretical_uph": 0,
            "predicted_uph": 0,
            "z_scores": z,
        }



def generate_scada_plot(report, save_path="process_health.png"):
    if report["z_scores"] is None:
        print(" Plot skipped: Emergency or OOD state.")
        return

    features = ["Temperature", "Pressure", "Vibration", "Speed"]
    z = report["z_scores"]

    colors = [
        "#2ecc71" if abs(v) <= 2 else "#f39c12" if abs(v) <= 3 else "#e74c3c"
        for v in z
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(features, z, color=colors, edgecolor="black", alpha=0.85)

    plt.axhline(0, color="black")
    plt.axhline(3, color="red", linestyle="--")
    plt.axhline(-3, color="red", linestyle="--")

    plt.axhspan(-2, 2, color="green", alpha=0.1, label="Normal (±2σ)")
    plt.axhspan(-3, 3, color="orange", alpha=0.05, label="Warning (±3σ)")

    plt.ylim(-6, 6)
    plt.ylabel("Deviation (σ)")
    plt.title(f"Process Health | Mode: {report['mode']}")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f" SCADA plot saved: {save_path}")


 
if __name__ == "__main__":
    dss = ManufacturingDSS("manufacturing.csv")

    report = dss.analyze(
        temp=205,
        pressure=82,
        vibration=1.8,
        speed=1200,
    )

    print("\n--- DECISION SUPPORT OUTPUT ---")
    for k, v in report.items():
        print(f"{k:20}: {v}")

    generate_scada_plot(report)
    display(Image("process_health.png"))
