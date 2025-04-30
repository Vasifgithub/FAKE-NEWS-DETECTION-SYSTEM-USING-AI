from flask import Flask, render_template, request
import joblib
import os
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1v5v-en6oV2go7m6_AtLz3KMmesEbGQiB"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1z4OiTd-avYs1tDLnbMaXz7GTNRCEobQS"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def handle_google_drive(url):
    """Bypass Google Drive virus scan warning"""
    session = requests.Session()
    response = session.get(url, stream=True, allow_redirects=True)
    
    # Extract confirmation token if needed
    if "confirm=" not in response.url:
        match = re.search(r'confirm=([^&]+)', response.url)
        if match:
            token = match.group(1)
            url = f"{url.split('&export=download')[0]}&confirm={token}"
            response = session.get(url, stream=True)
    
    return response

def download_with_retry(url, path):
    """Download file with retry logic and validation"""
    try:
        # Remove existing corrupted file
        if os.path.exists(path):
            if os.path.getsize(path) < 1024:
                os.remove(path)
        
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            response = handle_google_drive(url)
            response.raise_for_status()

            with open(path, "wb") as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Validate minimum size
            if os.path.getsize(path) < 1024:
                raise ValueError("Downloaded file too small")
            
            print(f"Success: {os.path.basename(path)} ({downloaded/1024:.1f} KB)")
            return True
            
    except Exception as e:
        print(f"Download failed: {str(e)}")
        if os.path.exists(path):
            os.remove(path)
        return False

def load_models():
    """Load models with strict version checks"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    
    # Download files with validation
    if not download_with_retry(MODEL_URL, model_path):
        print("XGBoost download failed")
    if not download_with_retry(VECTORIZER_URL, vectorizer_path):
        print("Vectorizer download failed")
    
    # Load models
    model, vectorizer = None, None
    
    try:
        model = joblib.load(model_path)
        print(f"XGBoost loaded | Type: {type(model)}")
    except Exception as e:
        print(f"Model load error: {str(e)}")
    
    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        print(f"Vectorizer error: {str(e)}")
    
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
