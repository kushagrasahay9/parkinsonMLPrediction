
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from exceptions import find_exceptions, predict_with_exceptions
import joblib

print("✅ Starting Parkinson's Ensemble Training...")

# --------------------------
# 1. Load Dataset
# --------------------------
print("[STEP 1] Loading dataset...")

data = pd.read_csv("parkinsons_converted.csv")   # <<< IMPORTANT

# Drop name column if exists
if 'name' in data.columns:
    data = data.drop(columns=['name'])

LABEL_COL = 'status'  # your dataset label column

if LABEL_COL not in data.columns:
    raise ValueError(f"❌ Label column '{LABEL_COL}' not found in dataset! Open CSV & confirm column name.")

X = data.drop(columns=[LABEL_COL])
y = data[LABEL_COL]

# --------------------------
# 2. Scale data & Split
# --------------------------
print("[STEP 2] Preprocessing data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 3. Train Models
# --------------------------
print("[STEP 3] Training base models...")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

scores = {}

for name, model in models.items():
    print(f"🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    scores[name] = (acc, f1)
    print(f"✅ {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# --------------------------
# 4. Pick Top 3 Models
# --------------------------
sorted_models = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)
top_models = [m[0] for m in sorted_models[:3]]
print(f"\n🏆 Top Models Selected: {top_models}")

top_estimators = [(name, models[name]) for name in top_models]

# --------------------------
# 5. Build Voting Ensemble
# --------------------------
print("\n[STEP 4] Building Voting Ensemble...")

ensemble = VotingClassifier(estimators=top_estimators, voting='soft')
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)

acc_ens = accuracy_score(y_test, y_pred_ensemble)
f1_ens = f1_score(y_test, y_pred_ensemble)
print(f"📊 Ensemble: Accuracy={acc_ens:.4f}, F1={f1_ens:.4f}")

# --------------------------
# 6. Exception Handling
# --------------------------
print("\n[STEP 5] Applying Exception Handling...")

exceptions, exception_labels = find_exceptions(ensemble, X_train, y_train)
y_pred_ex = predict_with_exceptions(ensemble, X_test, exceptions, exception_labels)

acc_ex = accuracy_score(y_test, y_pred_ex)
f1_ex = f1_score(y_test, y_pred_ex)
print(f"🚨 Ensemble + Exception Handling: Accuracy={acc_ex:.4f}, F1={f1_ex:.4f}")

# --------------------------
# 7. Save Results
# --------------------------
os.makedirs("results", exist_ok=True)

# Confusion Matrix plot
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_ex), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (After Exception Handling)")
plt.savefig("results/confusion_matrix.png")
plt.close()

# Text reports
with open("results/metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Best Base Model: {sorted_models[0][0]}\n")
    f.write(f"Voting Ensemble: Accuracy={acc_ens:.4f}, F1={f1_ens:.4f}\n")
    f.write(f"Ensemble + Exceptions: Accuracy={acc_ex:.4f}, F1={f1_ex:.4f}\n")

# ✅ Fixed classification report block
with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(str(classification_report(y_test, y_pred_ex)))

# Save exception samples
exceptions_df = pd.DataFrame(exceptions, columns=X.columns)
exceptions_df["true_label"] = exception_labels
exceptions_df.to_csv("results/exceptions_list.csv", index=False)

# Save model + scaler
joblib.dump(ensemble, "results/final_model.joblib")
joblib.dump(scaler, "results/scaler.joblib")

print("\n✅ Training Completed!")
print("📁 Results saved in 'results/' folder")
print("➡️ Run: python predict_single.py to test with new data")
