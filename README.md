 Fake News Detection Using-AI
This project is a web-based Fake News Detection system that uses a pre-trained XGBoost model to classify whether a given news text is Real or Fake (Rumor).

Built using Flask, Bootstrap 5, and XGBoost, this app provides a simple and attractive user interface where users can paste a news article and get instant predictions.

📋 Features
🧠 Fake News detection using a high-accuracy XGBoost model.

✨ Clean and responsive UI designed with Bootstrap 5.

📜 Single model setup for consistent and reliable predictions.

⚡ Fast and lightweight Flask application.

🛠️ Tech Stack
Frontend: HTML5, CSS3, Bootstrap 5

Backend: Python (Flask)

Model: XGBoost Classifier (pre-trained)

Other Tools: Scikit-learn, Pandas, Numpy

🖥️ Screenshots

Home Page	Prediction Result
(You can add your real screenshots later)

⚙️ Installation and Setup
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
2. Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy
Edit
Flask
scikit-learn
xgboost
numpy
pandas
(You can generate it with pip freeze > requirements.txt)

4. Place the pre-trained model
Save your trained XGBoost model file (example: xgboost_model.pkl) inside the project directory.

5. Run the Flask App
bash
Copy
Edit
python app.py
The app will be running at http://127.0.0.1:5000

📁 Project Structure
csharp
Copy
Edit
fake-news-detection/
│
├── static/
│   └── style.css         # Custom CSS styles
│
├── templates/
│   └── index.html        # Frontend HTML page
│
├── app.py                # Flask application
├── xgboost_model.pkl      # Pre-trained XGBoost model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
✨ Future Improvements
Add model retraining feature with new datasets.

Show probability/confidence scores with prediction.

Deploy the app on platforms like Render, Vercel, or AWS.

🤝 Acknowledgements
XGBoost Official Documentation

Flask Documentation

Bootstrap Documentation

📜 License
This project is licensed under the MIT License.

🔥 Author
Wasif

Passionate about AI, Deep Learning, and building meaningful projects.
