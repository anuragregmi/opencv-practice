#!/usr/bin/env python

import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
DATA_DIR = os.path.dirname(os.path.abspath(cv2.data.__file__))


def main():
    face_cascade = cv2.CascadeClassifier(
        os.path.join(DATA_DIR, 'haarcascade_frontalface_alt2.xml'))
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = dict()
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")  # grayscale
                image_array = np.array(pil_image, 'uint8')

                faces = face_cascade.detectMultiScale(
                    image_array, scaleFactor=1.5, minNeighbors=3
                )
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("lables.pkl", "wb") as f:
        lable_ids = {v: k for k, v in label_ids.items()}
        pickle.dump(lable_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trained.yml")


if __name__ == "__main__":
    main()
