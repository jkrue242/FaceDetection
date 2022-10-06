import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
yLabels = []
xTrain = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # creating np.array out of image pixel data
            pilImage = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            finalImage = pilImage.resize(size, Image.Resampling.LANCZOS)
            imageArray = np.array(finalImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = imageArray[y:y + h, x:x + w]
                xTrain.append(roi)
                yLabels.append(id_)


with open("pickles/labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("recognizers/trainer.yml")
