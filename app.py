from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
from utils.face_detector import crop_face
import cv2
import time

from utils.forensic import analyze_image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.image_quality import image_quality

app = Flask(__name__)

# LOAD MODEL
model = load_model("model/model.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(filepath):
    face = crop_face(filepath)

    if face is None:
        raise Exception("No face detected in the uploaded image.")

    face = cv2.resize(face, (224, 224))
    img_array = face.astype("float32") / 255.0
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

    print(file.filename)
    print(filepath)

    try:

        start = time.time()

        prediction, confidence = predict_image(filepath)

        info = analyze_image(filepath)
        quality = image_quality(filepath)

        processing_time = round(time.time() - start, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

    return render_template(
      "result.html",
       prediction=prediction,
       confidence=round(confidence * 100, 2),
       filename=file.filename,
       info=info,
       quality=quality,
       processing_time=processing_time
    )
if __name__ == '__main__':
    app.run(debug=True)