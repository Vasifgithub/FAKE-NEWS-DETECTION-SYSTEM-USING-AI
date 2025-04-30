from flask import Flask, render_template, request
import joblib
import os
import requests
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def get_google_drive_file(url):
    """Handle Google Drive downloads with confirmation"""
    session = requests.Session()
    
    # Initial request to get cookies
    response = session.get(url, stream=True)
    response.raise_for_status()
    
    # Extract confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    
    # Make confirmed download URL
    if token:
        url = f"{url}&confirm={token}"
    
    return session.get(url, stream=True)

def download_file(url, path):
    """Robust download with retries and validation"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Remove existing corrupted files
        if os.path.exists(path):
            os.remove(path)
        
        print(f"Downloading {os.path.basename(path)}...")
        response = get_google_drive_file(url)
        response.raise_for_status()

        # Write file
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Validate file
        if os.path.getsize(path) < 1024:
            raise ValueError("File too small")
        
        # Quick load validation
        try:
            joblib.load(path)
        except Exception as e:
            print(f"File validation failed: {str(e)}")
            os.remove(path)
            return False
        
        print(f"Downloaded {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        if os.path.exists(path):
            os.remove(path)
        return False

def load_models():
    """Load models with version compatibility"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    # Download models with retries
    for _ in range(3):  # 3 retries
        if download_file(MODEL_URL, model_path) and \
           download_file(VECTORIZER_URL, vectorizer_path):
            break
    else:
        raise RuntimeError("Failed to download model files after 3 attempts")

    # Force numpy initialization
    _ = np.array([0])
    
    # Load models
    model, vectorizer = None, None
    try:
        model = joblib.load(model_path)
        print(f"XGBoost loaded | NumPy {np.__version__} | XGB {model.__class__.__module__}")
    except Exception as e:
        print(f"Model load error: {str(e)}")
        raise

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        print(f"Vectorizer error: {str(e)}")
        raise

    return model, vectorizer

# Initialize models at startup
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Text preprocessing with validation"""
    if text and tfidf_vectorizer:
        try:
            return tfidf_vectorizer.transform([text.strip()])
        except Exception as e:
            print(f"Text processing error: {str(e)}")
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    text_result = None
    text_input = ""
    error_message = None

    if request.method == "POST":
        text_input = request.form.get("text", "").strip()
        
        if not text_input:
            error_message = "Please enter text to analyze"
        else:
            try:
                processed_text = preprocess_text(text_input)
                if processed_text is not None:
                    prediction = model.predict_proba(processed_text)[0][1]
                    text_result = "TRUE NEWS" if prediction >= 0.5 else "FAKE NEWS"
                else:
                    error_message = "Text processing failed"
            except Exception as e:
                error_message = "Analysis error - please try again"
                print(f"Prediction error: {str(e)}")

    return render_template("index.html",
                         text_result=text_result,
                         text_input=text_input,
                         error_message=error_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
