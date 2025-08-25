import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference
import mlflow
import mlflow.sklearn
import joblib
import logging

# 📂 Define and create base project and model directory
base_dir = os.path.join("C:/Users/Admin", "credit")
model_dir = os.path.join(base_dir, "model")
reports_dir = os.path.join(base_dir, "reports")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# 🧮 Sample dataset
data = pd.DataFrame({
    "income": [35000, 42000, 52000, 29000, 61000, 48000, 33000, 58000, 44000, 39000],
    "credit_score": [650, 700, 720, 580, 760, 690, 640, 750, 710, 660],
    "gender": ["male", "female", "female", "male", "male", "female", "female", "male", "male", "female"],
    "employment_status": ["employed", "self-employed", "employed", "unemployed", "employed", "employed", "student", "self-employed", "employed", "student"],
    "loan_approved": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
})

# 🏷️ Encode categorical features
le_gender = LabelEncoder()
le_employment = LabelEncoder()
data["gender"] = le_gender.fit_transform(data["gender"])
data["employment_status"] = le_employment.fit_transform(data["employment_status"])

X = data.drop("loan_approved", axis=1)
y = data["loan_approved"]

# 📊 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🤖 Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Metrics
accuracy = accuracy_score(y_test, y_pred)
fairness = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test["gender"])

# 💾 Save model with joblib
local_model_path = os.path.join(model_dir, "model.joblib")
joblib.dump(model, local_model_path)

# 📋 Start MLflow tracking and log metrics + model
mlflow.set_experiment("loan-approval")
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("fairness", fairness)
    
    # Log the saved model as an artifact to MLflow
    mlflow.log_artifact(local_model_path, artifact_path="model")

# Save training report
report_path = os.path.join(reports_dir, "training_report.txt")
with open(report_path, "w") as f:
    f.write("Model Training Report\n")
    f.write("=====================\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Fairness (Demographic Parity Difference): {fairness:.4f}\n")

print(f"📄 Report saved to: {report_path}")

# Setup logging
log_path = os.path.join(reports_dir, "train_model.log")
logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)

logging.info(f"Model trained with accuracy: {accuracy:.4f}")
logging.info(f"Fairness metric: {fairness:.4f}")

print(f"✅ Model trained and saved locally in: {model_dir}")
print(f"📊 Accuracy: {accuracy:.2f}, Fairness: {fairness:.2f}")
print(f"📝 Logs saved to: {log_path}")
