from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load XGBoost model
try:
    model = joblib.load("models/XGBoost.pkl")
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    model = None

# Load TF-IDF vectorizer
try:
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
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

    if request.method == "POST":
        text_input = request.form.get("text")

        if text_input and model and tfidf_vectorizer:
            processed_text = preprocess_text(text_input)
            try:
                prediction = model.predict_proba(processed_text)[:, 1]
                text_result = "TRUE-NEWS" if prediction[0] > 0.5 else "FAKE-NEWS"
            except Exception as e:
                print(f"Prediction error: {e}")
                text_result = "Prediction Error"

    return render_template("index.html", text_result=text_result, text_input=text_input)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
