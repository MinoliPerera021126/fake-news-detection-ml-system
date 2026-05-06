# 🔍 Fake News Detection Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning-powered web application that automatically classifies news articles as **REAL** or **FAKE**. Built with a lightweight Flask backend and a modern, dark-themed, responsive frontend UI, this tool helps identify misinformation by analyzing lexical patterns and text features.

The full ML lifecycle — from experiment tracking and model comparison to hyperparameter tuning, model registry, and production promotion — is managed end-to-end with **MLflow**.

---

## ✨ Features

- **Real-Time Inference:** Instantly analyzes news titles and article content.
- **Modern Dashboard UI:** A sleek, non-scrollable dark mode interface built with CSS grid for a premium user experience.
- **Detailed Analytics:** Displays not just the final prediction, but also visual confidence bars and specific probability percentages for both classes.
- **High Accuracy Model:** Powered by a highly optimized Linear Support Vector Machine (SVM) deployed for fast CPU inference.
- **Full ML Lifecycle Tracking:** Every experiment, metric, artifact, and model version is logged and governed via MLflow.

---

## 🧠 Model Architecture & Performance

The underlying machine learning pipeline was trained on the **ISOT Fake News Dataset** and utilizes TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.

| Property | Value |
|---|---|
| Algorithm | Linear Support Vector Classifier (Calibrated for probabilities) |
| Feature Extraction | TF-IDF Vectorizer — Unigrams & Bigrams, Max Features: 50,000 |
| Input | News title + article body (concatenated, cleaned) |
| Test Accuracy | 99.71% |
| Test F1-Score (Macro) | 99.71% |
| Test ROC-AUC | 99.97% |

### Model Selection Process
Six candidate models were trained and compared on the validation set before Linear SVM was selected:

| Model | Val F1 (Macro) | Val ROC-AUC |
|---|---|---|
| Logistic Regression | 0.989 | 0.999 |
| Multinomial Naive Bayes | 0.962 | 0.992 |
| **Linear SVM** ✅ | **0.996** | **1.00** |
| Random Forest | 0.99 | 0.999 |
| XGBoost | 0.993 | 1.00 |
| 1D CNN | 0.991 | 1.00 |

---

## 🔬 MLflow Experiment Tracking

This project uses **MLflow** to manage the complete ML lifecycle across all pipeline stages.

### What is tracked

- **Per-model metrics** — Accuracy, Precision, Recall, F1-macro, ROC-AUC, and training time for all 6 candidate models
- **Artifacts** — Confusion matrices, classification reports, and ROC curves per model
- **Hyperparameter tuning** — Full `RandomizedSearchCV` CV results for Linear SVM (`C` sweep), best params, and tuning curve
- **Ablation study** — F1 and AUC scores across 4 feature configurations (Title only / Text only / Title+Text / Title+Text+Subject)
- **Final evaluation** — Test set metrics, F1 progression chart (Baseline → Tuned → Test), and probability distribution plot

### Model Registry lifecycle

The selected model (Linear SVM) is formally governed through three registry stages:
```text
v1  →  Staging     (default params, val-evaluated, pre-tuning baseline)
v2  →  Staging     (tuned params, post-RandomizedSearchCV)
v2  →  Production  (final, test-evaluated, Flask-ready)
```

### Viewing the MLflow UI locally

```bash
mlflow ui
```
Navigate to `http://127.0.0.1:5000` (or port 5001 if Flask is running simultaneously).
The experiment is named **`Fake News Detection`** and contains the following runs:

| Run Name | Stage |
|---|---|
| `LogisticRegression` | Model comparison |
| `NaiveBayes` | Model comparison |
| `LinearSVM` | Model comparison |
| `RandomForest` | Model comparison |
| `XGBoost` | Model comparison |
| `1D_CNN` | Model comparison |
| `LinearSVM_Baseline_Snapshot` | Pre-tuning reference |
| `LinearSVM_HyperparamTuning` | HP tuning |
| `LinearSVM_AblationStudy` | Feature ablation |
| `LinearSVM_FinalEvaluation` | Final test evaluation |

---

## 📂 Project Structure

```text
fake-news-detection-ml-system/
│
├── data/                            # Raw dataset (not committed — see note below)
│
├── mlruns/                          # MLflow tracking data (not committed)
│
├── models/
│   ├── best_model.pkl               # Trained Linear SVM — not committed (see note)
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF Vectorizer — not committed
│   └── model_metadata.json          # Model metadata and final metrics ✅
│
├── static/
│   └── style.css                    # Dark theme styling and layout
│
├── templates/
│   └── index.html                   # Frontend UI dashboard
│
├── fake_news_detection.ipynb        # Full ML pipeline (Steps 1–12 + inference test)
├── app.py                           # Main Flask application logic
├── requirements.txt
├── mlflow.db                        # Local MLflow SQLite backend (not committed)
├── .gitignore
└── README.md
```

> **Note on data and model files:** The raw dataset is not committed to this repository. Download the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place the CSV files inside `data/`. Trained model binaries (`.pkl`) are excluded via `.gitignore` — run the notebook end-to-end to regenerate them.

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/MinoliPerera021126/fake-news-detection-ml-system.git
cd fake-news-detection-ml-system
```

### 2. Create and activate a virtual environment
```bash
python -m venv myvenv
myvenv\Scripts\activate        # Windows
# source myvenv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Regenerate model files (first time only)
Open and run `fake_news_detection.ipynb` end-to-end. This trains and evaluates all models, runs the full MLflow pipeline, and exports `best_model.pkl` and `tfidf_vectorizer.pkl` to `models/`.

### 5. Run the Flask application
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` in your browser.

### 6. (Optional) Launch the MLflow UI
```bash
mlflow ui --port 5001
```
Navigate to `http://127.0.0.1:5001` to explore all experiment runs, metrics, artifacts, and the model registry.

---

## 👥 Authors

| Name                | CPM   | MC     |
| ------------------- | ----- | ------ |
| Minoli Perera       | 24375 | 108853 |
| Sandagomi Kodikara  | 24377 | 108852 |
| Savindi Dissanayake | 24381 | 108798 |
