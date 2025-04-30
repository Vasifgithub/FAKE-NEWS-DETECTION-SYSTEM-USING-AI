from flask import Flask, render_template, request
import joblib
import os
import requests
import re
import numpy as np  # Imported first for version control
from sklearn.feature_extraction.text import TfidfVectorizer

# Validate critical dependencies immediately
assert np.__version__ == '1.23.5', f"Invalid numpy version: {np.__version__}"

app = Flask(__name__)

# Configuration - Use direct download URLs (not Google Drive)
MODEL_URL = "https://your-domain.com/models/XGBoost.pkl"
VECTORIZER_URL = "https://your-domain.com/models/tfidf_vectorizer.pkl"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def handle_google_drive(url):
    """Proper Google Drive download handling with confirmation"""
    session = requests.Session()
    
    # Initial request to get cookies
    response = session.get(url, stream=True)
    response.raise_for_status()
    
    # Extract confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    
    # Build confirmed URL
    if token:
        url = f"{url}&confirm={token}"
    
    return session.get(url, stream=True)

def download_file(url, path):
    """Secure file download with validation"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Remove existing file
        if os.path.exists(path):
            os.remove(path)
        
        print(f"Downloading {os.path.basename(path)}...")
        response = handle_google_drive(url)
        response.raise_for_status()

        # Write file
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Validate file
        if os.path.getsize(path) < 1024:
            raise ValueError("File too small")
        
        # Load test
        _ = joblib.load(path)
        
        print(f"Downloaded {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"Download failed: {str(e)}")
        if os.path.exists(path):
            os.remove(path)
        return False

def load_models():
    """Load models with version validation"""
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    
    # Download with 3 retries
    for attempt in range(3):
        if download_file(MODEL_URL, model_path) and \
           download_file(VECTORIZER_URL, vectorizer_path):
            break
        print(f"Retry {attempt + 1}/3")
    else:
        raise RuntimeError("Failed to download models after 3 attempts")

    # Force numpy initialization
    _ = np.array([0])
    
    # Load models
    try:
        model = joblib.load(model_path)
        print(f"XGBoost loaded | XGBoost {model.__class__.__module__}")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {str(e)}")

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        raise RuntimeError(f"Vectorizer load failed: {str(e)}")

    return model, vectorizer

# Initialize models
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Text preprocessing with error handling"""
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
