from PIL import Image
import os

def analyze_image(filepath):

    img = Image.open(filepath)

    width, height = img.size

    extension = os.path.splitext(filepath)[1].upper().replace(".", "")

    size = os.path.getsize(filepath) / 1024

    report = []

    # Resolution
    if width >= 1000 and height >= 1000:
        report.append("High Resolution")
    else:
        report.append("Low Resolution")

    # Image size
    if size < 150:
        report.append("High Compression")
    else:
        report.append("Normal Compression")

    # Face
    report.append("Face Detected")

    return {
        "width": width,
        "height": height,
        "format": extension,
        "size": round(size, 2),
        "report": report
    }