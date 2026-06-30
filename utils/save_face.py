import cv2
import os
from utils.face_detector import crop_face

def save_face(image_path, filename):

    face = crop_face(image_path)

    if face is None:
        return None

    output = os.path.join(
        "static",
        "results",
        filename
    )

    cv2.imwrite(output, face)

    return "results/" + filename