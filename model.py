import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATA ----------------
# Excel file load
df = pd.read_csv("dataset.csv")

# ---------------- COLUMN CHECK ----------------
print(df.head())

# Expected Columns:
# Vibration (mm/s)
# Temperature (°C)
# Pressure (bar)
# RMS Vibration
# Mean Temp
# Fault (Target)

# ---------------- DATA CLEANING ----------------
df = df.dropna()

# ---------------- FEATURES & TARGET ----------------
X = df[['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)',
        'RMS Vibration', 'Mean Temp']]

y = df['Fault Label']   # 0 = Good, 1 = Bad, 2 = Moderate

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved as model.pkl ✅")
