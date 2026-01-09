import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Daily Appliance Prediction", layout="wide")
st.title("🔌 Daily Appliance Power: Actual vs Predicted (Live)")

# -----------------------------------
# Load model
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/power_regressor.pkl")

regressor = load_model()

# -----------------------------------
# Upload CSV
# -----------------------------------
uploaded_file = st.file_uploader("Upload appliance CSV", type=["csv"])

if uploaded_file:
    df_iter = pd.read_csv(uploaded_file, chunksize=1)
    st.success("CSV uploaded. Processing live...")

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
    buffers = {a: [] for a in appliance_cols}

    current_day = None
    day_data = {}

    container = st.empty()   # 🔥 dynamic UI container

    for chunk in df_iter:
        row = chunk.iloc[0]
        ts = pd.to_datetime(row["timeStamp"])
        row_day = ts.date()

        # Initialize first day
        if current_day is None:
            current_day = row_day
            day_data = {a: {"t": [], "act": [], "pred": []} for a in appliance_cols}

        # Day change detected → SHOW UI IMMEDIATELY
        if row_day != current_day:
            with container.container():
                st.subheader(f"📅 {current_day}")
                for app in appliance_cols:
                    if day_data[app]["act"]:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.plot(day_data[app]["t"], day_data[app]["act"],
                                color="red", label="Actual")
                        ax.plot(day_data[app]["t"], day_data[app]["pred"],
                                color="blue", label="Predicted")
                        ax.set_title(app)
                        ax.legend()
                        st.pyplot(fig)

            # Reset for next day
            current_day = row_day
            day_data = {a: {"t": [], "act": [], "pred": []} for a in appliance_cols}

        # Process row (streaming)
        for app in appliance_cols:
            val = row[app]
            buffers[app].append(val)
            if len(buffers[app]) > WINDOW:
                buffers[app].pop(0)

            if len(buffers[app]) < WINDOW:
                continue

            window = np.array(buffers[app])

            if val == 0:
                pred = 0
            else:
                features = pd.DataFrame([{
                    "Mean_Power": window.mean(),
                    "Std_Power": window.std(),
                    "Max_Power": window.max(),
                    "Min_Power": window.min(),
                    "Power_Delta": window[-1] - window[0],
                    "On_Ratio": (window > 0).sum() / WINDOW
                }])
                pred = regressor.predict(features)[0]

            day_data[app]["t"].append(ts)
            day_data[app]["act"].append(val)
            day_data[app]["pred"].append(pred)

        time.sleep(0.01)  # smooth UI update

    # -----------------------------------
    # SHOW LAST DAY
    # -----------------------------------
    with container.container():
        st.subheader(f"📅 {current_day}")
        for app in appliance_cols:
            if day_data[app]["act"]:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(day_data[app]["t"], day_data[app]["act"],
                        color="red", label="Actual")
                ax.plot(day_data[app]["t"], day_data[app]["pred"],
                        color="blue", label="Predicted")
                ax.set_title(app)
                ax.legend()
                st.pyplot(fig)

    st.success("✅ All days processed and displayed")
