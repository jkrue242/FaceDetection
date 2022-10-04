# Written by Joseph Krueger

import cv2
import numpy as np

haarCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

# camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # detecting face in grayscale
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = haarCascade.detectMultiScale(grayImg, scaleFactor=1.5, minNeighbors=5)

    for x, y, w, h in face:

        # determine roi
        roiGray = grayImg[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]

        # create rectangle
        rectColor = (255, 100, 0)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectColor, thickness)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
