import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('model/model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize the image to match model input size
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # If the filename is not valid, we generate a new one
        filename = secure_filename(file.filename)
        if not filename or not (filename.endswith('.jpeg') or filename.endswith('.jpg')):
            filename = 'captured_image.jpeg'
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        img = preprocess_image(filepath)
        os.remove(filepath)
        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        max_probability = np.max(predictions)

        print(f"Predictions: {predictions}")
        print(f"Predicted class: {predicted_class}")
        print(f"Max probability: {max_probability}")

        # Confidence threshold
        confidence_threshold = 0.8  # Adjust this value as needed

        # Map the predicted class index to the corresponding label
        if max_probability < confidence_threshold:
            return jsonify({'prediction': 'Show a proper picture that depicts the American sign language'})
        else:
            if predicted_class < 10:
                predicted_label = str(predicted_class)  # Numbers from 0 to 9
            else:
                predicted_label = chr(predicted_class + ord('a') - 10)  # Alphabets from 'a' to 'z'
            
            return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
