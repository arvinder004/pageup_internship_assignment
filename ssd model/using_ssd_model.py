import cv2
import face_recognition
import pickle
import csv
from datetime import datetime
import os

ENCODINGS_FILE = "face_encodings_ssd.pkl"
LOG_FILE = "recognized_faces_ssd_log.csv"
PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Loaded {len(known_face_names)} known faces.")
else:
    known_face_encodings = []
    known_face_names = []

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "First Seen", "Last Seen"])

face_occurrences = {}

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            face_frame = frame[startY:endY, startX:endX]
            rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(rgb_face)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                else:
                    name = input("Enter the name of the new face: ")
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)

                    with open(ENCODINGS_FILE, "wb") as file:
                        pickle.dump((known_face_encodings, known_face_names), file)
                    print(f"Saved new face: {name}")

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if name not in face_occurrences:
                    face_occurrences[name] = {"first_seen": current_time, "last_seen": current_time}
                else:
                    face_occurrences[name]["last_seen"] = current_time

                with open(LOG_FILE, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Name", "First Seen", "Last Seen"])
                    for name, occurrences in face_occurrences.items():
                        writer.writerow([name, occurrences["first_seen"], occurrences["last_seen"]])

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("SSD Face Detection", frame)

video_capture.release()
cv2.destroyAllWindows()