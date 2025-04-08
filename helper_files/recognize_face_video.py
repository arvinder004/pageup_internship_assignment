# Takes a video input and recognizes faces using pre-trained encodings and saves the data of first occurance and last occurance in a .csv file

import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ENCODINGS_FILE = "face_encodings.pkl"
LOG_FILE = "recognized_faces_log.csv"

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Loaded {len(known_face_names)} known faces.")
else:
    print("No known faces found. Please save faces first.")
    exit()

face_occurrences = {}

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "First Seen", "Last Seen"])

video_path = 'video2.mp4'
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or error reading frame.")
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

            if name not in face_occurrences:
                face_occurrences[name] = {"first_seen": current_time, "last_seen": current_time}
            else:
                face_occurrences[name]["last_seen"] = current_time

        top, right, bottom, left = [v * 2 for v in face_location]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Detecting faces from video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

with open(LOG_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "First Seen", "Last Seen"])
    for name, occurrences in face_occurrences.items():
        writer.writerow([name, occurrences["first_seen"], occurrences["last_seen"]])

video_capture.release()
cv2.destroyAllWindows()
