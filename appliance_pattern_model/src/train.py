import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "pattern_data.csv"))

FEATURES = [
    "Mean_Power",
    "Std_Power",
    "Max_Power",
    "Min_Power",
    "Power_Delta",
    "On_Ratio"
]

X = df[FEATURES]
y_class = df["Appliance"]
y_power = df["Actual_Power"]

X.replace([np.inf, -np.inf], 0, inplace=True)
X.fillna(0, inplace=True)

X_train, X_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 🔹 Classifier (pattern → appliance)
classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)
classifier.fit(X_train, yc_train)

print("Classifier accuracy:", accuracy_score(yc_test, classifier.predict(X_test)))

# 🔹 Regressor (pattern → power)
regressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
regressor.fit(X, y_power)

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(classifier, os.path.join(BASE_DIR, "models", "pattern_classifier.pkl"))
joblib.dump(regressor, os.path.join(BASE_DIR, "models", "power_regressor.pkl"))

print("✅ Models trained and saved")
