# MediPredict – AI in Healthcare

A simple Flask web app with two AI tools:
- **Diabetes risk prediction**: a Random Forest trained on the Pima Indians Diabetes dataset (768 records). Test accuracy is 77.27%.
- **Brain tumor MRI classification**: a CNN that sorts scans into glioma, meningioma, pituitary or no tumor. Test accuracy is 56.09%.

> For learning and demonstration only. Not a medical diagnosis tool.

## How to run

1. Clone. The brain model is stored with Git LFS, so run `git lfs install` before cloning.
   ```
   git clone https://github.com/austinantonyroy/AI-Based-Multi-Disease-Medical-Screening-and-Prediction-System.git
   ```
2. Install: `pip install -r requirements.txt`
3. Train: `python train.py`. You only need to do this once; it creates `model.pkl`.
4. Run: `python app.py`, then open http://127.0.0.1:5000
5. Evaluate: `python evaluate.py`. This prints accuracy, precision, recall, F1-score and the confusion matrix for both models. The brain part needs the MRI `Testing` folder; set its path in `TEST_DIR`.

## Pages
- **Home**: overview of both tools.
- **Diabetes**: choose a gender and enter the clinical values to get a risk % (Low, Moderate or High). Pregnancies is hidden for males.
- **Brain MRI**: upload an MRI scan to get the predicted tumor type and the confidence for each class.

## Files
| File | Purpose |
|---|---|
| `diabetes.csv` | Pima Indians Diabetes dataset |
| `train.py` | Trains the diabetes Random Forest and saves `model.pkl` |
| `model.pkl` | Trained diabetes model |
| `brain_tumor_model.h5` | Trained brain tumor CNN (Git LFS) |
| `evaluate.py` | Evaluation metrics for both models |
| `app.py` | Flask web app (routes and predictions) |
| `templates/` | HTML pages |
| `static/style.css` | Design and styling |

## Datasets
- Pima Indians Diabetes Database: Smith et al., 1988 (UCI ML Repository / Kaggle)
- Brain Tumor Classification (MRI): Bhuvaji et al., Kaggle, 2020
