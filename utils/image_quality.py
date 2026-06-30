import cv2
import numpy as np

def image_quality(path):

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur detection
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    if sharpness > 150:
        sharp = "High"
    elif sharpness > 60:
        sharp = "Medium"
    else:
        sharp = "Low"

    brightness = np.mean(gray)

    if brightness < 60:
        light = "Dark"

    elif brightness > 180:
        light = "Over Exposed"

    else:
        light = "Normal"

    return {
        "sharpness": sharp,
        "brightness": light
    }