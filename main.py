# Written by Joseph Krueger

from train import *

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(recognizerFilePath)

# reading from pickle file
with open(labelsPicklePath, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # running cascade on grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    for (x, y, w, h) in faces:

        # setting roi
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # recognizing
        id_, conf = recognizer.predict(roi_gray)

        if 10 <= conf <= 90:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y-10), font, 1, color, stroke, cv2.LINE_AA)

        # drawing rectangle around face
        color = (255, 255, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release capture when finished
cap.release()
cv2.destroyAllWindows()
