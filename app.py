from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load XGBoost model
try:
    model_path = os.path.join(MODEL_DIR, "models/XGBoost.pkl")
    model = joblib.load(model_path)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    model = None

# Load TF-IDF vectorizer
try:
    vectorizer_path = os.path.join(MODEL_DIR, "models/tfidf_vectorizer.pkl")
    with open(vectorizer_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF Vectorizer: {e}")
    tfidf_vectorizer = None

def preprocess_text(text):
    if tfidf_vectorizer:
        return tfidf_vectorizer.transform([text])
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    text_result = None
    text_input = ""
    error_message = None

    if request.method == "POST":
        text_input = request.form.get("text")

        if not text_input:
            error_message = "Please enter some text to analyze."
        elif not model or not tfidf_vectorizer:
            error_message = "Model initialization error - please try again later."
        else:
            try:
                processed_text = preprocess_text(text_input)
                if processed_text is not None:
                    prediction = model.predict_proba(processed_text)[:, 1]
                    text_result = "TRUE-NEWS" if prediction[0] > 0.5 else "FAKE-NEWS"
                else:
                    error_message = "Text processing failed."
            except Exception as e:
                print(f"Prediction error: {e}")
                error_message = "Analysis failed - please try again."

    return render_template("index.html", 
                         text_result=text_result,
                         text_input=text_input,
                         error_message=error_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
