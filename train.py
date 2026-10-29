"""Train a simple diabetes model and save it to model.pkl."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("diabetes.csv")

# Zeros in these columns mean "missing", so replace them with the median
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    data[col] = data[col].replace(0, data[col].median())

X = data.drop("Outcome", axis=1)  # features
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train.values, y_train)

print(f"Test accuracy: {model.score(X_test.values, y_test):.2%}")
joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")
print("Saved model.pkl")
