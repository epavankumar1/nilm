import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clf = joblib.load(os.path.join(BASE_DIR, "models", "pattern_classifier.pkl"))
reg = joblib.load(os.path.join(BASE_DIR, "models", "power_regressor.pkl"))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", r"C:\appliance_pattern_model\data\raw\applianceStatus_102595148_28Mar_31Mar.csv"))
df["timeStamp"] = pd.to_datetime(df["timeStamp"])
df = df.sort_values("timeStamp")
df.fillna(0, inplace=True)

# Example appliance to test
APPLIANCE = "Refrigerator"
WINDOW = 5

actual = []
predicted = []
timestamps = []

values = df[APPLIANCE].values
times = df["timeStamp"].values

for i in range(len(values) - WINDOW):
    window = values[i:i+WINDOW]
    if window.sum() == 0:
        continue

    features = pd.DataFrame([{
        "Mean_Power": window.mean(),
        "Std_Power": window.std(),
        "Max_Power": window.max(),
        "Min_Power": window.min(),
        "Power_Delta": window[-1] - window[0],
        "On_Ratio": (window > 0).sum() / WINDOW
    }])

    appliance_pred = clf.predict(features)[0]
    power_pred = reg.predict(features)[0]

    if appliance_pred == APPLIANCE:
        actual.append(window.mean())
        predicted.append(power_pred)
        timestamps.append(times[i+WINDOW])

# Plot
plt.figure()
plt.plot(timestamps, actual, label="Actual Power", linewidth=2)
plt.plot(timestamps, predicted, label="Predicted Power", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.title(f"Actual vs Predicted Power – {APPLIANCE}")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
