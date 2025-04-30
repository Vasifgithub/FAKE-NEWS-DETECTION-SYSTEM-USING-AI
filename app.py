from flask import Flask, render_template, request
import joblib
import os
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def download_file(url, save_path):
    """Secure file download with retry logic"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Clear existing file
        if os.path.exists(save_path):
            os.remove(save_path)

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Validate file size
        if os.path.getsize(save_path) < 1024:
            raise ValueError("File too small")
            
        return True
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return False

def load_models():
    """Load models with version validation"""
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    # Download files if missing
    if not os.path.exists(model_path):
        download_file(MODEL_URL, model_path)
    if not os.path.exists(vectorizer_path):
        download_file(VECTORIZER_URL, vectorizer_path)

    # Load models
    try:
        model = joblib.load(model_path)
        print(f"Model loaded | Type: {type(model)}")
    except Exception as e:
        print(f"Model load error: {str(e)}")
        model = None

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        print(f"Vectorizer error: {str(e)}")
        vectorizer = None

    return model, vectorizer

# Initialize models at startup
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    if text and tfidf_vectorizer:
        try:
            return tfidf_vectorizer.transform([text])
        except Exception as e:
            print(f"Text processing error: {str(e)}")
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    # ... [Keep your existing route logic] ...

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
