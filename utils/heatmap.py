import cv2
import numpy as np
import os

def generate_heatmap(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    heat = cv2.GaussianBlur(gray, (55,55), 0)

    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heat,0.4,0)

    save_path = image_path.replace(
        "uploads",
        "results"
    ).rsplit(".",1)[0] + "_heatmap.jpg"

    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    cv2.imwrite(save_path,overlay)

    return os.path.basename(save_path)