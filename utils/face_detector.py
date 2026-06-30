import cv2

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # If no face is found, use the entire image
    if len(faces) == 0:
        return img

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + pad * 2)
    h = min(img.shape[0] - y, h + pad * 2)

    return img[y:y+h, x:x+w]