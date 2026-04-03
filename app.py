from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and vectorizer from the models folder using joblib
model_path = os.path.join('models', 'best_model.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Route for the main home page
@app.route('/')
def home():
    return render_template('index.html')

# Route that handles the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the UI form
        news_title = request.form.get('title', '')
        news_text = request.form.get('text', '')
        
        # Combine just like in the Jupyter Notebook
        combined_text = news_title + ' ' + news_text
        
        if not combined_text.strip():
            return jsonify({'error': 'Please enter some text to analyze.'})

        # Transform and predict
        vec = vectorizer.transform([combined_text])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]

        # Format the results
        result = 'REAL' if prediction == 1 else 'FAKE'
        confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            'label': result,
            'confidence': confidence,
            'fake_prob': round(float(probabilities[0]) * 100, 2),
            'real_prob': round(float(probabilities[1]) * 100, 2)
        })

if __name__ == '__main__':
    app.run(debug=True)