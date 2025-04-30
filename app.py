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

def download_model_files():
    """Download required model files if missing"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    files = {
        "XGBoost.pkl": MODEL_URL,
        "tfidf_vectorizer.pkl": VECTORIZER_URL
    }

    for filename, url in files.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")

def load_models():
    """Load ML models with error handling"""
    download_model_files()
    
    model, vectorizer = None, None
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "XGBoost.pkl"))
        print("XGBoost model loaded successfully")
    except Exception as e:
        print(f"Error loading XGBoost model: {str(e)}")

    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        print("TF-IDF vectorizer loaded successfully")
    except Exception as e:
        print(f"Error loading TF-IDF vectorizer: {str(e)}")

    return model, vectorizer

# Initialize models at startup
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Transform input text using the vectorizer"""
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
            error_message = "Please enter some text to analyze"
        elif not model or not tfidf_vectorizer:
            error_message = "Model initialization error - please try again later"
        else:
            try:
                processed_text = preprocess_text(text_input)
                if processed_text is not None:
                    prediction = model.predict_proba(processed_text)[0][1]
                    text_result = "TRUE-NEWS" if prediction >= 0.5 else "FAKE-NEWS"
                else:
                    error_message = "Text processing failed"
            except Exception as e:
                error_message = "Analysis error - please try again"
                print(f"Prediction error: {str(e)}")

    return render_template("index.html",
                         text_result=text_result,
                         text_input=text_input,
                         error_message=error_message)

# Properly indented main block
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
