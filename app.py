from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the image model (for image prediction)
image_model = load_model('Fake_News_Detection.h5')

# Load the text model (for text prediction)
text_model = load_model('lstm_model.h5')
label_encoder = LabelEncoder()  # Assuming labels are encoded for text prediction

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_input_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Change size based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def preprocess_input_text(text):
    # Preprocess the text input (tokenization, padding, etc. - modify this based on your text model requirements)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([text])
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)  # Adjust maxlen based on your model's requirement
    return padded

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve files from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process Image Input
        image_file = request.files.get('file')
        text_input = request.form.get('text')

        image_result = None
        text_result = None
        uploaded_image_url = None

        # Handle image input
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            # Preprocess the image and predict
            img = preprocess_input_image(filepath)
            image_prediction = image_model.predict(img)

            # Assuming the model predicts a binary classification (adjust threshold accordingly)
            image_result = "Rumor" if image_prediction[0][0] > 0.8 else "Non-Rumor"
            uploaded_image_url = url_for('uploaded_file', filename=filename)

        # Handle text input
        elif text_input:
            text_input_processed = preprocess_input_text(text_input)
            text_prediction = text_model.predict(text_input_processed)

            # Assuming the model predicts a binary classification (adjust threshold accordingly)
            text_result = "Rumor" if text_prediction[0] > 0.5 else "Non-Rumor"

        # Return the result to the frontend
        return render_template('index.html', image_result=image_result, text_result=text_result,
                               uploaded_image_url=uploaded_image_url, text_input=text_input)

    return render_template('index.html', image_result=None, text_result=None)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, port=8080)
