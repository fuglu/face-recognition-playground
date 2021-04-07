import pickle
from pathlib import Path

import cv2
import face_recognition

SCALE_PERCENT = 0.25

encodings = []
names = []
images = []

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
    result = face_recognition.face_encodings(image)
    if (len(result) > 0):
        encoding = face_recognition.face_encodings(image)[0]
        encodings.append(encoding)

        # Get name from filename
        names.append(file.stem)

        images.append(file)

pickle.dump(encodings, open("cache/encodings.pickle", "wb"))
pickle.dump(names, open("cache/names.pickle", "wb"))
pickle.dump(images, open("cache/images.pickle", "wb"))
