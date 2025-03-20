from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf

try:
    text_clf = joblib.load("text_classification_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Text classifier and vectorizer loaded successfully!")
except Exception as e:
    print("Error loading text classifier or vectorizer:", e)
    raise

text_class_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def preprocess_text(text):
    """
    Preprocess the input text using the loaded TF-IDF vectorizer.
    This returns the same 10,000-dimension feature vector used during training.
    """
    vector = vectorizer.transform([text])
    return vector


try:
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    image_model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    image_model.load_weights('final_model.weights.h5')
    print("Image classifier model loaded successfully!")
except Exception as e:
    print("Error loading image classifier:", e)
    raise

def preprocess_image(image_data):
    """
    Preprocess the image input.
    Expects image_data as a base64-encoded string (with or without a data URI header).
    Converts it to a 224x224 numpy array normalized to [0,1].
    """
    try:
        if image_data.startswith("data:image"):
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        return img_array
    except Exception as e:
        print("Error during image preprocessing:", e)
        raise


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json(force=True)
        input_type = data.get("type", "text")  
        
        if input_type == "text":
            text = data.get("text", "")
            if not text:
                return jsonify({"error": "No text provided"}), 400
            
            processed = preprocess_text(text)
            preds = text_clf.predict(processed)
            print("Raw text prediction output:", preds)
            
            # For multi-label, iterate over each label prediction
            pred_array = preds[0] 
            predicted_labels = []
            for i, val in enumerate(pred_array):
                if val == 1:
                    predicted_labels.append(text_class_labels[i])
            if not predicted_labels:
                predicted_labels = ["totally fine"]
            return jsonify({"label": ", ".join(predicted_labels)})
        
        elif input_type == "image":
            image_data = data.get("image", "")
            if not image_data:
                return jsonify({"error": "No image provided"}), 400
            
            processed_image = preprocess_image(image_data)
            pred = image_model.predict(processed_image)
            print("Raw image prediction output:", pred)
            # For binary classification: output >= 0.5 means "sensitive/harmful"
            predicted_label = "sensitive/harmful" if pred[0][0] >= 0.5 else "not sensitive/harmful"
            return jsonify({"label": predicted_label})
        
        else:
            return jsonify({"error": "Invalid input type. Use 'text' or 'image'."}), 400
    except Exception as e:
        print("Error during classification:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
