import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", r"C:\appliance_pattern_model\data\raw\applianceStatus_102595148_28Mar_31Mar.csv"))
df["timeStamp"] = pd.to_datetime(df["timeStamp"])
df = df.sort_values("timeStamp")
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

WINDOW = 5   # pattern window (rows)

records = []

for app in appliance_cols:
    values = df[app].values

    for i in range(len(values) - WINDOW):
        window = values[i:i+WINDOW]

        if window.sum() == 0:
            continue

        records.append({
            "Appliance": app,
            "Mean_Power": window.mean(),
            "Std_Power": window.std(),
            "Max_Power": window.max(),
            "Min_Power": window.min(),
            "Power_Delta": window[-1] - window[0],
            "On_Ratio": (window > 0).sum() / WINDOW,
            "Actual_Power": window.mean()   # target for regression
        })

pattern_df = pd.DataFrame(records)

out_path = os.path.join(BASE_DIR, "data", "processed", "pattern_data.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pattern_df.to_csv(out_path, index=False)

print("✅ Pattern dataset created")
