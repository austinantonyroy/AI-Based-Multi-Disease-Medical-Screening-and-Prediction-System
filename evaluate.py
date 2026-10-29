"""Evaluate both models: accuracy, precision, recall, F1-score and confusion matrix."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TensorFlow info/warning messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import glob
import logging
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------- 1. Diabetes model (Random Forest) ----------
data = pd.read_csv("diabetes.csv")
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    data[col] = data[col].replace(0, data[col].median())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Same split as train.py, so these are the 154 test records the model never saw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("model.pkl")["model"]
y_pred = model.predict(X_test.values)

print("=" * 50)
print("DIABETES MODEL (Random Forest)")
print("=" * 50)
print(f"Test records: {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred), "\n")
print(classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))

print("Feature importance:")
for name, score in sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:<26}{score:.3f}")

# ---------- 2. Brain tumor model (CNN) ----------
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
TEST_DIR = "../asher medical/Testing"  # 394 test MRI images, one folder per class

print("\nChecking 394 MRI test images, please wait about 1 minute...")
logging.getLogger("absl").setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model
brain_model = load_model("brain_tumor_model.h5")

y_true, y_pred = [], []
for label, cls in enumerate(CLASSES):
    for path in glob.glob(f"{TEST_DIR}/{cls}/*"):
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = np.array(img) / 255.0
        y_pred.append(int(np.argmax(brain_model.predict(x[np.newaxis], verbose=0))))
        y_true.append(label)

print("\n" + "=" * 50)
print("BRAIN TUMOR MODEL (CNN)")
print("=" * 50)
print(f"Test images: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}\n")
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_true, y_pred), "\n")
print(classification_report(y_true, y_pred, target_names=["Glioma", "Meningioma", "No tumor", "Pituitary"]))
