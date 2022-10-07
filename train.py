import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cId = 0
labelIds = {}
yLabels = []
xTrain = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if not label in labelIds:
                labelIds[label] = cId
                cId += 1
            id_ = labelIds[label]

            # convert image to numpy array
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(final_image, "uint8")

            faces = faceCascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                xTrain.append(roi)
                yLabels.append(id_)

with open("pickles/labels.pickle", 'wb') as f:
    pickle.dump(labelIds, f)

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("recognizers/trainer.yml")
