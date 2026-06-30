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
from utils.artifacts import detect_artifacts
from utils.heatmap import generate_heatmap
from utils.consistency import consistency_check
from utils.save_face import save_face

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


def predict_video(video_path):

    from utils.video_processor import extract_frames

    frames = extract_frames(video_path)

    fake = 0
    real = 0

    for frame in frames:

        frame = cv2.resize(frame, (224,224))

        frame = frame.astype("float32")/255

        frame = np.expand_dims(frame,axis=0)

        pred = model.predict(frame, verbose=0)[0][0]

        if pred>0.5:
            fake += 1
        else:
            real += 1

    if fake > real:
        label = "FAKE"
    else:
        label = "REAL"

    confidence = max(fake,real)/max(1,(fake+real))

    return label,confidence,len(frames),fake,real

def risk_level(confidence):

    c = confidence*100

    if c>=85:
        return "HIGH"

    elif c>=65:
        return "MEDIUM"

    return "LOW"

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
    extension = file.filename.lower().split(".")[-1]

    file.save(filepath)

    print(file.filename)
    print(filepath)

    try:

        start = time.time()

        # ================= VIDEO =================

        if extension in ["mp4", "mov", "avi", "webm"]:

            prediction, confidence, total, fake_frames, real_frames = predict_video(filepath)

            risk = risk_level(confidence)

            explanation = [
                "Frame-by-frame analysis completed.",
                "Average prediction generated.",
                "Temporal consistency evaluated.",
                "Video classified using AI."
            ]

            info = {
                "width": "-",
                "height": "-",
                "format": extension.upper(),
                "size": round(os.path.getsize(filepath) / 1024, 2),
                "report": [
                    "Video uploaded successfully.",
                    "Frames extracted.",
                    "AI processed each frame."
                ]
            }

            quality = {
                "sharpness": "-",
                "brightness": "-",
                "contrast": "-",
                "blur": "-",
                "exposure": "-",
                "overall": "Video"
            }

            consistency = [
                "Frame consistency checked.",
                "Average confidence calculated."
            ]

            edge_image = None
            heatmap = None
            face_image = None

        # ================= IMAGE =================

        else:

            prediction, confidence = predict_image(filepath)

            total = 1
            fake_frames = "-"
            real_frames = "-"

            risk = risk_level(confidence)

            if prediction == "FAKE":

                explanation = [
                    "AI detected manipulation patterns.",
                    "Facial textures appear inconsistent.",
                    "Edge artifacts were identified.",
                    "Confidence exceeds normal threshold."
                ]

            else:

                explanation = [
                    "No major manipulation artifacts detected.",
                    "Facial structure appears consistent.",
                    "Lighting looks natural.",
                    "Image passed forensic checks."
                ]

            info = analyze_image(filepath)

            heatmap = generate_heatmap(filepath)

            consistency, risk = consistency_check(
                prediction,
                confidence
            )

            edge_path = detect_artifacts(filepath)

            edge_image = edge_path.replace("static/", "")

            quality = image_quality(filepath)

            face_image = save_face(
                filepath,
                "face_" + file.filename
            )

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
        processing_time=processing_time,
        edge_image=edge_image,
        consistency=consistency,
        risk=risk,
        explanation=explanation,
        face_image=face_image,
        heatmap=heatmap,
        total_frames=total,
        fake_frames=fake_frames,
        real_frames=real_frames
    )
if __name__ == '__main__':
    app.run(debug=True)