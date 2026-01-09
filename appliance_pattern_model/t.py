import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "appliance_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "power_regressor.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "predicted_vs_actual.csv")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -----------------------------
# Load model ONCE
# -----------------------------
regressor = joblib.load(MODEL_PATH)

# -----------------------------
# Load CSV (FAST)
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["timeStamp"] = pd.to_datetime(df["timeStamp"])
df.fillna(0, inplace=True)

appliance_cols = [
    "Refrigerator",
    "Air conditioner",
    "Washing machine",
    "Geyser",
    "TV",
    "Mixie",
    "Air cooler"
]

WINDOW = 5
rows = []

# -----------------------------
# FAST BATCH PROCESSING
# -----------------------------
for appliance in appliance_cols:
    values = df[appliance].values
    times = df["timeStamp"].values

    if values.sum() == 0:
        continue

    feature_rows = []
    meta = []

    for i in range(len(values) - WINDOW):
        window = values[i:i + WINDOW]
        actual = values[i + WINDOW - 1]

        feature_rows.append([
            window.mean(),
            window.std(),
            window.max(),
            window.min(),
            window[-1] - window[0],
            (window > 0).sum() / WINDOW
        ])

        meta.append((times[i + WINDOW - 1], appliance, actual))

    # 🚀 ONE SINGLE PREDICT CALL
    features_df = pd.DataFrame(
        feature_rows,
        columns=[
            "Mean_Power",
            "Std_Power",
            "Max_Power",
            "Min_Power",
            "Power_Delta",
            "On_Ratio"
        ]
    )

    preds = regressor.predict(features_df)

    for (ts, app, actual), pred in zip(meta, preds):
        rows.append({
            "timeStamp": ts,
            "Appliance": app,
            "Actual_Power": round(actual, 2),
            "Predicted_Power": round(pred, 2),
            "Error": round(pred - actual, 2)
        })

# -----------------------------
# Save output
# -----------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Prediction completed in seconds")
print(f"📄 Output saved to: {OUTPUT_PATH}")
