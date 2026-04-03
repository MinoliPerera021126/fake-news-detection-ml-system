# 🔍 Fake News Detection Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning-powered web application that automatically classifies news articles as **REAL** or **FAKE**. Built with a lightweight Flask backend and a modern, dark-themed, responsive frontend UI, this tool helps identify misinformation by analyzing lexical patterns and text features.

## ✨ Features

- **Real-Time Inference:** Instantly analyzes news titles and article content.
- **Modern Dashboard UI:** A sleek, non-scrollable dark mode interface built with CSS grid for a premium user experience.
- **Detailed Analytics:** Displays not just the final prediction, but also visual confidence bars and specific probability percentages for both classes.
- **High Accuracy Model:** Powered by a highly optimized Linear Support Vector Machine (SVM) deployed for fast CPU inference.

## 🧠 Model Architecture & Performance

The underlying machine learning pipeline was trained on the **ISOT Fake News Dataset** and utilizes TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.

- **Algorithm:** Linear Support Vector Classifier (Calibrated for probabilities)
- **Feature Extraction:** TF-IDF Vectorizer (Unigrams & Bigrams, Max Features: 50,000)
- **Test Accuracy:** 99.71%
- **Test F1-Score (Macro):** 99.71%
- **Validation ROC-AUC:** 99.97%

## 📂 Project Structure

```text
fake_news_app/
│
├── models/
│   ├── best_model.pkl          # Trained Linear SVM model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer
│
├── static/
│   └── style.css               # Dark theme styling and layout
│
├── templates/
│   └── index.html              # Frontend UI dashboard
│
├── app.py                      # Main Flask application logic
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Setup

To run this application locally on your machine, follow these steps:

1. Create a virtual environment

```bash
python -m venv venv
```

2. Activate the virtual environment

```bash
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Access the web app
   Open your web browser and navigate to: `http://127.0.0.1:5000`

---

## 👥 Authors

| Name                | CPM   | MC     |
| ------------------- | ----- | ------ |
| Minoli Perera       | 24375 | 108853 |
| Sandagomi Kodikara  | 24377 | 108852 |
| Savindi Dissanayake | 24381 | 108798 |
