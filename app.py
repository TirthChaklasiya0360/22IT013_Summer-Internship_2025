import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model for Devanagari characters (ensure the path is correct)
model = load_model('devanagari_character_cnn.h5')  # Update the model filename if necessary

# Define allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Mapping of class numbers (0-45) to Devanagari characters
devanagari_characters = [
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',  # Class 0-9
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',  # Class 10-19
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',  # Class 20-29
    'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ','0','1','2','3',       # Class 30-39
    '4','5','6','7','8','9'                              #class 40-45
]

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    img_url = None
    predicted_character = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded image
            upload_path = os.path.join('static', 'uploaded_image.png')  # Path to save the uploaded image
            file.save(upload_path)  # Save the file to the static folder
            img_url = url_for('static', filename='uploaded_image.png')

            # Process the image and make a prediction
            img = Image.open(upload_path)
            img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
            img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Reshape and normalize

            # Predict the Devanagari character class
            prediction = np.argmax(model.predict(img), axis=-1)[0]

            # Get the corresponding Devanagari character
            predicted_character = devanagari_characters[prediction]

    return render_template('index.html', prediction=predicted_character, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True) 
    
