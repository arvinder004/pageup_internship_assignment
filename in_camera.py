import cv2
import pickle
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import csv
import os
from datetime import datetime

video = cv2.VideoCapture(0)

with open('data/face_data.pkl', 'rb') as f:
    face_data = pickle.load(f)

known_names = []
known_encodings = []

for name, encodings in face_data.items():
    for encoding in encodings:
        known_names.append(name)
        known_encodings.append(encoding)

print("Number of known names:", len(known_names))
print("Number of known encodings:", len(known_encodings))

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(known_encodings, known_names)

DISTANCE_THRESHOLD = 0.4
CSV_FILE = "emp_attendance.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "In Time", "Out Time", "Duration (seconds)"])

def read_attendance():
    attendance = {}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = row["Name"]
                in_time = row["In Time"]
                if in_time:
                    attendance[name] = {"in_time": in_time}
    return attendance

attendance = read_attendance()

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_time = datetime.now()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)

        if min_distance < DISTANCE_THRESHOLD:
            match_index = np.argmin(distances)
            name = known_names[match_index]

            if name not in attendance:
                attendance[name] = {"in_time": current_time.strftime("%Y-%m-%d %H:%M:%S")}
                with open(CSV_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S"), "", ""])
            else:
                print(f"{name} already has an in-time entry. Skipping...")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("In Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()