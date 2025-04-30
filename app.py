from flask import Flask, render_template, request
import joblib
import os
import requests
import re
import numpy as np  # Explicit import with version check
from sklearn.feature_extraction.text import TfidfVectorizer

# Check numpy version early
try:
    assert np.__version__ == '1.23.5'
except (AssertionError, ImportError) as e:
    print(f"CRITICAL: Wrong numpy version ({np.__version__}) - must be 1.23.5")
    raise

app = Flask(__name__)

# Configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def handle_google_drive(url):
    """Bypass Google Drive virus scan confirmation"""
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Extract confirmation token if needed
    if "confirm=" not in response.url:
        match = re.search(r'confirm=([^&]+)', response.url)
        if match:
            new_url = f"{url}&confirm={match.group(1)}"
            response = session.get(new_url, stream=True)
    
    return response

def validate_model_file(path):
    """Check if file is a valid pickle"""
    try:
        with open(path, "rb") as f:
            joblib.load(f)
        return True
    except Exception as e:
        print(f"Invalid model file: {str(e)}")
        return False

def download_model_file(url, path):
    """Secure download with validation"""
    try:
        # Remove existing invalid files
        if os.path.exists(path):
            if os.path.getsize(path) < 1024 or not validate_model_file(path):
                os.remove(path)
        
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            response = handle_google_drive(url)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Post-download validation
            if not validate_model_file(path):
                raise ValueError("Downloaded file failed validation")
            
            print(f"Success: {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"Download failed: {str(e)}")
        if os.path.exists(path):
            os.remove(path)
        return False

def load_models():
    """Load models with numpy compatibility checks"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    # Download models
    download_model_file(MODEL_URL, model_path)
    download_model_file(VECTORIZER_URL, vectorizer_path)

    # Initialize numpy first
    _ = np.array([0])  # Force numpy initialization
    
    # Load models
    model, vectorizer = None, None
    
    try:
        model = joblib.load(model_path)
        print(f"XGBoost loaded | NumPy {np.__version__} | XGBoost {model.__class__.__module__}")
    except Exception as e:
        print(f"Model load error: {str(e)}")

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        print(f"Vectorizer error: {str(e)}")

    return model, vectorizer

# Initialize models
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Text processing with numpy compatibility"""
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
