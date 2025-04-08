import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ENCODINGS_FILE = "face_encodings.pkl"
LOG_FILE = "first_seen_time.csv"

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Loaded {len(known_face_names)} known faces.")
else:
    print("No known faces found. Please save faces first.")
    known_face_encodings = []
    known_face_names = []

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "First Seen"])

first_seen = {}

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name != "Unknown":
            if name not in first_seen:
                first_seen[name] = current_time

                with open(LOG_FILE, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Name", "First Seen"])
                    for name, time in first_seen.items():
                        writer.writerow([name, time])

        top, right, bottom, left = [v * 2 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("IN CAMERA", frame)

camera.release()
cv2.destroyAllWindows()