# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load your saved scikit-learn model (make sure this file is in your repository)
model = joblib.load("text_classification_model.pkl")
print("Model loaded successfully!")

# Define class labels (order must match your training)
class_labels = ["totally fine", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def preprocess_text(text):
    # Replace this dummy function with your actual preprocessing (e.g., TF-IDF transformation)
    # For demonstration, we return a random vector with shape (1, 100)
    return np.random.rand(1, 100)

app = Flask(__name__)
CORS(app)  # Enable CORS so that external clients can access your API

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess_text(text)
    preds = model.predict(processed)
    class_idx = np.argmax(preds, axis=1)[0] if preds.ndim > 1 else np.argmax(preds)
    predicted_label = class_labels[class_idx]

    return jsonify({"label": predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
