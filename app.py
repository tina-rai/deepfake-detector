from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# LOAD MODEL
model = load_model("model/model.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    print("RAW PREDICTION:", prediction)

    confidence = float(prediction)

    if confidence > 0.5:
        label = "FAKE"
        final_conf = confidence
    else:
        label = "REAL"
        final_conf = 1 - confidence

    return label, final_conf

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    #  DEBUG HERE
    print(file.filename)
    print(filepath)

    prediction, confidence = predict_image(filepath)

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        filename=file.filename
    )
if __name__ == '__main__':
    app.run(debug=True)
