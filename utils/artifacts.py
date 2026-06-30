import cv2
import os

def detect_artifacts(filepath):

    img = cv2.imread(filepath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    save_path = filepath.replace(
        "uploads",
        "results"
    ).rsplit(".",1)[0] + "_edges.jpg"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, edges)

    return save_path