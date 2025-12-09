import joblib
import numpy as np
import pandas as pd
import random

print("🔍 Loading model and scaler...")
scaler = joblib.load("results/scaler.joblib")
model = joblib.load("results/final_model.joblib")

print("📂 Reading dataset...")
data = pd.read_csv("parkinsons_converted.csv")

# Drop possible ID/name columns
for col in ['name', 'id', 'ID']:
    if col in data.columns:
        data = data.drop(columns=[col])

# Detect label column automatically
possible_labels = ['status', 'Status', 'target', 'class', 'Class']
label_col = None
for lbl in possible_labels:
    if lbl in data.columns:
        label_col = lbl
        break

if label_col is None:
    print("❌ Could not find label column. Columns available:", list(data.columns))
    exit()

X = data.drop(columns=[label_col])
print(f"✅ Detected label column: '{label_col}'")
print(f"Dataset shape: {X.shape}")

if X.shape[0] == 0:
    print("❌ Dataset has 0 rows — please verify your CSV.")
    exit()

# -------------------------------
# 🎲 PICK A RANDOM SAMPLE
# -------------------------------
rand_index = random.randint(0, len(X) - 1)
sample = X.iloc[rand_index].to_numpy().reshape(1, -1)
print(f"\n🎲 Random Sample Selected (Row {rand_index}):")
print(X.iloc[rand_index].to_string())

# -------------------------------
# 🔮 MAKE PREDICTION
# -------------------------------
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
prob = model.predict_proba(sample_scaled)

print("\n🎯 Prediction Result:")
print("----------------------")
print(f"Predicted Class: {int(pred[0])} (0 = Healthy, 1 = Parkinson's)")
print(f"Confidence (Probabilities): {prob[0]}")
if pred[0] == 1:
    print("\n🧠 The model predicts this person likely has Parkinson’s Disease.")
else:
    print("\n💪 The model predicts this person is Healthy.")

print("\n✅ Test completed successfully.")
