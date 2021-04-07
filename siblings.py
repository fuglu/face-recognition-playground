import os
import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np

SCALE_PERCENT = 0.25

encodings = pickle.load(open("cache/encodings.pickle", "rb"))
names = pickle.load(open("cache/names.pickle", "rb"))
images = pickle.load(open("cache/images.pickle", "rb"))


def convert_image(file):
    image = cv2.imread(str(file), cv2.IMREAD_COLOR)

    # Rescale
    image = cv2.resize(image, (0, 0), fx=SCALE_PERCENT, fy=SCALE_PERCENT)

    # Convert to RGB (opencv users BGR and face_recognition uses RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_face_distances(file):

    image = convert_image(file)

    # Load a test image and get encondings for it
    image_encodings = face_recognition.face_encodings(image)
    if (len(image_encodings) > 0):
        image_to_test_encoding = face_recognition.face_encodings(image)[0]

        # See how far apart the test image is from the known faces
        face_distances = face_recognition.face_distance(
            encodings, image_to_test_encoding)

        return face_distances

    else:
        return None


p = Path("images")
for file in p.iterdir():
    print(f"Processing {file}")
    face_distances = get_face_distances(file)
    if (face_distances is not None):
        idx = np.argpartition(face_distances, 1)
        name = names[idx[1]]
        image = images[idx[1]]

        orig = cv2.imread(str(file), cv2.IMREAD_COLOR)
        orig = cv2.resize(orig, (0, 0), fx=SCALE_PERCENT, fy=SCALE_PERCENT)

        sibling = cv2.imread(str(image), cv2.IMREAD_COLOR)
        sibling = cv2.resize(
            sibling, (0, 0), fx=SCALE_PERCENT, fy=SCALE_PERCENT)

        cv2.imshow("Original", orig)
        cv2.imshow("Sibling", sibling)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
