import cv2

def cnvt_clr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]
