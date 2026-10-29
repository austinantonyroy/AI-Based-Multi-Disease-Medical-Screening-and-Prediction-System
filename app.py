"""Simple web app: diabetes risk prediction + brain tumor MRI classification."""
import base64
import io
import joblib
import numpy as np
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)

saved = joblib.load("model.pkl")
diabetes_model = saved["model"]

# Form fields shown on the diabetes page, in the order the model expects
FIELDS = [
    {"name": "Pregnancies", "label": "Pregnancies", "unit": "count", "placeholder": "0", "hint": "Number of times pregnant (0 if none)", "optional": True},
    {"name": "Glucose", "label": "Glucose", "unit": "mg/dL", "placeholder": "e.g. 120", "hint": "2-hour plasma glucose (normal < 140)"},
    {"name": "BloodPressure", "label": "Blood Pressure", "unit": "mm Hg", "placeholder": "e.g. 72", "hint": "Diastolic (lower) value"},
    {"name": "SkinThickness", "label": "Skin Thickness", "unit": "mm", "placeholder": "e.g. 29", "hint": "Triceps skin-fold thickness"},
    {"name": "Insulin", "label": "Insulin", "unit": "µU/mL", "placeholder": "0 if unknown", "hint": "2-hour serum insulin", "optional": True},
    {"name": "BMI", "label": "BMI", "unit": "kg/m²", "placeholder": "e.g. 26.5", "hint": "Body mass index"},
    {"name": "DiabetesPedigreeFunction", "label": "Family History Score", "unit": "DPF", "placeholder": "e.g. 0.47", "hint": "Diabetes pedigree function (0.1 – 2.5)"},
    {"name": "Age", "label": "Age", "unit": "years", "placeholder": "e.g. 35", "hint": "Age in years"},
]

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
CLASS_INFO = {
    "glioma_tumor": "A tumor that starts in the glial cells of the brain or spine.",
    "meningioma_tumor": "A usually slow-growing tumor of the membranes around the brain.",
    "no_tumor": "No tumor pattern was detected in this scan.",
    "pituitary_tumor": "A growth in the pituitary gland at the base of the brain.",
}
brain_model = None  # loaded on first use (TensorFlow is slow to import)


def risk_notes(v):
    """Simple warning signs based on common clinical thresholds."""
    notes = []
    if v["Glucose"] >= 140:
        notes.append("Glucose is above the normal range (≥ 140 mg/dL).")
    if v["BMI"] >= 30:
        notes.append("BMI is in the obese range (≥ 30).")
    elif v["BMI"] >= 25:
        notes.append("BMI is in the overweight range (25 – 30).")
    if v["BloodPressure"] >= 90:
        notes.append("Diastolic blood pressure is high (≥ 90 mm Hg).")
    if v["Age"] >= 45:
        notes.append("Age 45+ is a known risk factor.")
    return notes


@app.route("/")
def home():
    return render_template("home.html", page="home")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        values = {f["name"]: float(request.form.get(f["name"]) or 0) for f in FIELDS}
        if request.form.get("Gender") == "male":
            values["Pregnancies"] = 0
        chance = diabetes_model.predict_proba([list(values.values())])[0][1]
        if chance >= 0.5:
            level = "high"
        elif chance >= 0.3:
            level = "moderate"
        else:
            level = "low"
        result = {"percent": round(chance * 100), "level": level, "notes": risk_notes(values)}
    return render_template("diabetes.html", page="diabetes", fields=FIELDS, result=result)


@app.route("/brain", methods=["GET", "POST"])
def brain():
    global brain_model
    result = None
    if request.method == "POST":
        if brain_model is None:
            from tensorflow.keras.models import load_model
            brain_model = load_model("brain_tumor_model.h5")
        file = request.files["image"]
        data = file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = np.array(img.resize((224, 224))) / 255.0
        pred = brain_model.predict(x[np.newaxis], verbose=0)[0]
        best = CLASSES[int(np.argmax(pred))]
        result = {
            "key": best,
            "name": best.replace("_", " ").title(),
            "confidence": round(float(pred.max()) * 100, 1),
            "info": CLASS_INFO[best],
            "probs": [(c.replace("_", " ").title(), round(float(p) * 100, 1)) for c, p in zip(CLASSES, pred)],
            "image": f"data:{file.mimetype};base64," +base64.b64encode(data).decode(),
        }
    return render_template("brain.html", page="brain", result=result)


if __name__ == "__main__":
    app.run(debug=True)
