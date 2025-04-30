from flask import Flask, render_template, request
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Google Drive direct download links (keep .pkl extension)
MODEL_URL = "https://drive.google.com/file/d/1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB/view?usp=sharin"
VECTORIZER_URL = "https://drive.google.com/file/d/1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS/view?usp=sharing"

def download_file(url, save_path):
    """Robust Google Drive downloader with virus scan handling"""
    try:
        session = requests.Session()
        print(f"Starting download: {os.path.basename(save_path)}")
        
        # Initial request
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Handle Google Drive virus scan warning
        if "confirm=" not in response.url and "download_warning" in response.text:
            confirm_link = f"{url.split('&export=download')[0]}&confirm=t"
            response = session.get(confirm_link, stream=True)
            response.raise_for_status()

        # Write file with progress
        file_size = int(response.headers.get('Content-Length', 0))
        chunk_size = 8192
        downloaded = 0
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                downloaded += len(chunk)
                f.write(chunk)
                
        # Validate minimum size
        if downloaded < 1024:
            raise ValueError(f"File too small ({downloaded} bytes)")
            
        print(f"Downloaded {os.path.basename(save_path)} ({downloaded/1024:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"Download failed: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def load_models():
    """Load models with version compatibility checks"""
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    # Clean previous downloads
    for path in [model_path, vectorizer_path]:
        if os.path.exists(path):
            print(f"Removing existing: {os.path.basename(path)}")
            os.remove(path)

    # Download files
    if not download_file(MODEL_URL, model_path):
        print("XGBoost download failed")
        return None, None
        
    if not download_file(VECTORIZER_URL, vectorizer_path):
        print("Vectorizer download failed")
        return None, None

    # Load models with version validation
    try:
        model = joblib.load(model_path)
        print(f"XGBoost loaded | XGBoost version: {model.__class__.__module__}")
    except Exception as e:
        print(f"Model load failed: {str(e)}")
        model = None

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded | Features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        print(f"Vectorizer load failed: {str(e)}")
        vectorizer = None

    return model, vectorizer

# Initialize models at startup
model, tfidf_vectorizer = load_models()

def preprocess_text(text):
    """Text preprocessing with error handling"""
    if not text or not tfidf_vectorizer:
        return None
    try:
        return tfidf_vectorizer.transform([text.strip()])
    except Exception as e:
        print(f"Text processing error: {str(e)}")
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
            error_message = "System error - models not loaded"
        else:
            try:
                processed_text = preprocess_text(text_input)
                if processed_text is not None:
                    prediction = model.predict_proba(processed_text)[0][1]
                    text_result = "TRUE NEWS" if prediction >= 0.5 else "FAKE NEWS"
                else:
                    error_message = "Failed to process text"
            except Exception as e:
                error_message = "Analysis error - please try again"
                print(f"Prediction error: {str(e)}")

    return render_template("index.html",
                         text_result=text_result,
                         text_input=text_input,
                         error_message=error_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
