from flask import Flask, render_template, request
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Replace with ACTUAL direct download URLs
MODEL_URL = "https://drive.google.com/file/d/1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB/view?usp=sharin"
VECTORIZER_URL = "https://drive.google.com/file/d/1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS/view?usp=sharing"

def download_file(url, save_path):
    """Secure file download with validation"""
    try:
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    
        # Validate file size
        if os.path.getsize(save_path) < 1024:  # Minimum 1KB check
            raise ValueError("Downloaded file is too small - likely invalid")
            
        print(f"Success: {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"Download failed: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def load_models():
    """Robust model loading with validation"""
    model_path = os.path.join(MODEL_DIR, "XGBoost.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

    # Clean existing files
    for path in [model_path, vectorizer_path]:
        if os.path.exists(path):
            print(f"Removing potentially corrupted file: {path}")
            os.remove(path)

    # Download fresh copies
    if not download_file(MODEL_URL, model_path):
        return None, None
    if not download_file(VECTORIZER_URL, vectorizer_path):
        return None, None

    # Validate files
    try:
        model = joblib.load(model_path)
        print(f"XGBoost model loaded | Type: {type(model)}")
    except Exception as e:
        print(f"Model load error: {str(e)}")
        model = None

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Vocab size: {len(vectorizer.vocabulary_)}")
    except Exception as e:
        print(f"Vectorizer load error: {str(e)}")
        vectorizer = None

    return model, vectorizer

# Initialize models
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Safe text processing"""
    if not tfidf_vectorizer or not text:
        return None
    try:
        return tfidf_vectorizer.transform([text.strip()])
    except Exception as e:
        print(f"Text processing failed: {str(e)}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    text_result = None
    error_message = None
    text_input = request.form.get("text", "") if request.method == "POST" else ""

    if request.method == "POST":
        if not text_input.strip():
            error_message = "Please enter text to analyze"
        elif not model or not tfidf_vectorizer:
            error_message = "System error: Models not loaded"
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
