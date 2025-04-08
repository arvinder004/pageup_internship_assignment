import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ENCODINGS_FILE = "face_encodings.pkl"
LOG_FILE = "employee_attendance_log.csv"

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Loaded {len(known_face_names)} known faces")
else:
    print("No known faces found")
    exit()

face_occurences = {}

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "In Time"])

capture = cv2.VideoCapture(0)
print("press q to exit")

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error loading frame")
        break

    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encodings, face_locations in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if name not in face_occurences:
                face_occurences[name] = {"first_seen": current_time}
            
            with open(LOG_FILE, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "In Time"])
                for name, occurrences in face_occurences.items():
                    writer.writerow([name, occurrences["first_seen"]])
            
            top, right, bottom, left = [v*2 for v in face_locations]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 3)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Detecting Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
            