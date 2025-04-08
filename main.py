import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ENCODINGS_FILE = "face_encodings.pkl"
LOG_FILE = "in_out_time.csv"

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
        writer.writerow(["Name", "First Seen (In-Time)", "Last Seen (Out-Time)"])

first_seen = {}
last_seen = {}

in_camera = cv2.VideoCapture(0) 
out_camera = cv2.VideoCapture(1)  
if not in_camera.isOpened():
    print("Error: Could not open the IN camera.")
    exit()
if not out_camera.isOpened():
    print("Error: Could not open the OUT camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret_in, frame_in = in_camera.read()
    ret_out, frame_out = out_camera.read()

    if not ret_in:
        print("Error: Could not read frame from the IN camera.")
        continue
    if not ret_out:
        print("Error: Could not read frame from the OUT camera.")
        continue

    small_frame_in = cv2.resize(frame_in, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame_in = cv2.cvtColor(small_frame_in, cv2.COLOR_BGR2RGB)

    small_frame_out = cv2.resize(frame_out, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame_out = cv2.cvtColor(small_frame_out, cv2.COLOR_BGR2RGB)

    in_face_locations = face_recognition.face_locations(rgb_small_frame_in, model="cnn")
    in_face_encodings = face_recognition.face_encodings(rgb_small_frame_in, in_face_locations)

    out_face_locations = face_recognition.face_locations(rgb_small_frame_out, model="cnn")
    out_face_encodings = face_recognition.face_encodings(rgb_small_frame_out, out_face_locations)

    for in_face_encoding, in_face_location in zip(in_face_encodings, in_face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, in_face_encoding)
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
                    writer.writerow(["Name", "First Seen (In-Time)", "Last Seen (Out-Time)"])
                    for name in first_seen.keys():
                        writer.writerow([name, first_seen[name], last_seen.get(name, "")])

        top, right, bottom, left = [v * 2 for v in in_face_location]
        cv2.rectangle(frame_in, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame_in, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    for out_face_encoding, out_face_location in zip(out_face_encodings, out_face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, out_face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name != "Unknown":
            last_seen[name] = current_time

            with open(LOG_FILE, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "First Seen (In-Time)", "Last Seen (Out-Time)"])
                for name in first_seen.keys():
                    writer.writerow([name, first_seen[name], last_seen.get(name, "")])

        top, right, bottom, left = [v * 2 for v in out_face_location]
        cv2.rectangle(frame_out, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(frame_out, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("IN CAMERA", frame_in)
    cv2.imshow("OUT CAMERA", frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

in_camera.release()
out_camera.release()
cv2.destroyAllWindows()