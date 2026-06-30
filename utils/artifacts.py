import cv2

def detect_artifacts(path):

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,100,200)

    output = path.replace(".", "_edges.")

    cv2.imwrite(output, edges)

    return output