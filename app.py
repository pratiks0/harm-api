from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained classifier and TF-IDF vectorizer
try:
    clf = joblib.load("text_classification_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print("Error loading model or vectorizer:", e)
    raise

# Define class labels (order must match training)
class_labels = ["totally fine", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def preprocess_text(text):
    """
    Preprocess the input text using the loaded TF-IDF vectorizer.
    This converts the raw text into the same 10,000-dimension vector used during training.
    """
    # Transform the text. This returns a sparse matrix.
    vector = vectorizer.transform([text])
    return vector

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins explicitly
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        # Log the incoming request
        print("Received /classify request.")
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            print("No text provided in the request.")
            return jsonify({"error": "No text provided"}), 400

        # Preprocess the text to get the correct feature vector
        processed = preprocess_text(text)
        print("Processed text into feature vector of shape:", processed.shape)
        
        # Get predictions from the classifier
        preds = clf.predict(processed)
        print("Raw prediction output:", preds)
        
        # Determine the predicted class index
        if preds.ndim > 1:
            class_idx = np.argmax(preds, axis=1)[0]
        else:
            class_idx = np.argmax(preds)
        predicted_label = class_labels[class_idx]
        print("Predicted label:", predicted_label)
        
        # Return the prediction as JSON with CORS headers
        return jsonify({"label": predicted_label})
    except Exception as e:
        print("Error during classification:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app with debug mode enabled for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
