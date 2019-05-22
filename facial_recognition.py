#!/usr/bin/env python
import os
import cv2
import numpy as np
import pickle

DATA_DIR = os.path.dirname(os.path.abspath(cv2.data.__file__))

labels = dict()
with open("lables.pkl", "rb") as f:
    label = pickle.load(f)


def get_label_from_id(id_):
    return label.get(id_)


def main():
    face_cascade = cv2.CascadeClassifier(
        os.path.join(DATA_DIR, 'haarcascade_frontalface_alt2.xml'))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained.yml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=10)

        for i, (x, y, w, h) in enumerate(faces):
            # region of interest
            roi = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]

            # predict
            id_, conf = recognizer.predict(roi_gray)
            if 85 >= conf > 45:
                label = get_label_from_id(id_)
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                stroke = 2
                cv2.putText(frame, label, (x, y-10), font,
                            1, color, stroke, cv2.LINE_AA)

            cv2.imshow(f"Your Face {i}", roi)

            color = (255, 0, 0)  # BGR
            stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

        cv2.imshow('frame', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
