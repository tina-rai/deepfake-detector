import cv2
import numpy as np

def image_quality(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness
    sharpness = round(cv2.Laplacian(gray, cv2.CV_64F).var(),2)

    # Brightness
    brightness = round(np.mean(gray),2)

    # Contrast
    contrast = round(np.std(gray),2)

    # Blur score
    blur = "Low"

    if sharpness < 80:
        blur = "High"
    elif sharpness < 150:
        blur = "Medium"

    # Exposure

    if brightness < 70:
        exposure = "Under Exposed"

    elif brightness > 180:
        exposure = "Over Exposed"

    else:
        exposure = "Normal"

    # Overall Quality

    score = 100

    if sharpness < 120:
        score -= 20

    if contrast < 40:
        score -= 20

    if brightness < 60 or brightness > 190:
        score -= 20

    if score >= 80:
        overall = "Excellent"

    elif score >= 60:
        overall = "Good"

    else:
        overall = "Poor"

    return {

        "sharpness": sharpness,

        "brightness": brightness,

        "contrast": contrast,

        "blur": blur,

        "exposure": exposure,

        "overall": overall

    }