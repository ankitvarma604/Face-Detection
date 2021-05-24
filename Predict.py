import numpy as np
import cv2
import os
import face_detect as fd
import training_data_prepare as td

faces, labels = td.prepare_training_data('Images/')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

subjects = ["", "Ranbir", "Elvis Presley","Change it with your name"]


def predict(test_img):
    img = test_img.copy()
    face, rect = fd.detect_face(img)
    label = recognizer.predict(face)
    label_text = subjects[label[0]]
    td.draw_rectangle(img, rect)
    td.draw_text(img, label_text, rect[0], rect[1] - 2)
    return img



print("Predicting images...")