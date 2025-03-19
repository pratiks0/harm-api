from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained classifier and TF-IDF vectorizer
try:
    # Example: a multi-label classifier
    clf = joblib.load("text_classification_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print("Error loading model or vectorizer:", e)
    raise

# Define class labels (DO NOT include "totally fine" here)
# We'll detect "totally fine" if all predictions are 0
class_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def preprocess_text(text):
    """
    Preprocess the input text using the loaded TF-IDF vectorizer.
    This converts the raw text into the same feature vector used during training.
    """
    # Transform the text. This returns a sparse matrix
    vector = vectorizer.transform([text])
    return vector

app = Flask(__name__)
# Enable CORS for all routes/origins
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        print("Received /classify request.")
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            print("No text provided in the request.")
            return jsonify({"error": "No text provided"}), 400

        # Convert raw text to TF-IDF features
        processed = preprocess_text(text)
        print("Processed text into feature vector of shape:", processed.shape)
        
        # Predict with the multi-label classifier
        # For multi-label, clf.predict(...) typically returns something like [[0 1 0 1 0 ...]]
        preds = clf.predict(processed)
        print("Raw prediction output:", preds)

        # preds is an array of shape (1, N) if you have N classes
        # e.g. preds might be [[1 0 1 0 0]]
        pred_array = preds[0]

        # Convert 0/1 array into a list of labels
        predicted_labels = []
        for i, val in enumerate(pred_array):
            if val == 1:
                predicted_labels.append(class_labels[i])

        # If none are 1, interpret as "totally fine"
        if not predicted_labels:
            predicted_labels = ["totally fine"]

        predicted_label_str = ", ".join(predicted_labels)
        print("Predicted label(s):", predicted_label_str)

        return jsonify({"label": predicted_label_str})
    except Exception as e:
        print("Error during classification:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
