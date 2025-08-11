from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model once when the app starts
model = tf.keras.models.load_model('model3.hdf5', compile=False)

# Classes
classes = {0: 'Normal Tire', 1: 'Cracked Tire'}
THRESHOLD = 0.5

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((300, 300), Image.LANCZOS)
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # channel
    img = np.expand_dims(img, axis=0)   # batch
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = 1 if prediction[0][0] >= THRESHOLD else 0
    return classes[class_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file selected", image_url=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected", image_url=None)
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        prediction = predict_image(filepath)
        image_url = filepath

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
