import os
from pathlib import Path

import cv2
import face_recognition

SCALE_PERCENT = 0.25

# Open webcam
video_capture = cv2.VideoCapture(0)

# Process image folder and collect known faces
encodings = []
names = []

p = Path("images")
for file in p.iterdir():
    print(f"Processing {file}")

    # Read image
    image = cv2.imread(str(file), cv2.IMREAD_COLOR)

    # Rescale
    image = cv2.resize(image, (0, 0), fx=SCALE_PERCENT, fy=SCALE_PERCENT)

    # Convert to RGB (opencv users BGR and face_recognition uses RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Uncomment to debug image
    # cv2.imshow(file.stem, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Encode image
    encoding = face_recognition.face_encodings(image)[0]
    encodings.append(encoding)

    # Get name from filename
    names.append(file.stem)


while True:
    # Get next frame from webcam
    ret, frame = video_capture.read()

    # Rescale frame
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_PERCENT, fy=SCALE_PERCENT)

    # Convert to RGB (opencv users BGR and face_recognition uses RGB)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations
    face_locations = face_recognition.face_locations(small_frame)

    # Encode all faces
    face_encodings = face_recognition.face_encodings(
        small_frame, face_locations)

    # Collect names by checking each detected face with known faces
    found_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(encodings, face_encoding)

        for i, match in enumerate(matches):
            if match:
                found_names.append(names[i])

    # Draw rectangle around faces
    for (top, right, bottom, left), name in zip(face_locations, found_names):
        # Undo scale
        top *= int(1/SCALE_PERCENT)
        right *= int(1/SCALE_PERCENT)
        bottom *= int(1/SCALE_PERCENT)
        left *= int(1/SCALE_PERCENT)

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw name
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit any key to quit
    key = cv2.waitKey(1)
    if key > 0:
        break

# Release webcam
video_capture.release()
cv2.destroyAllWindows()
