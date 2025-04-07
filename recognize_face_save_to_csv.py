import cv2
import face_recognition
import os
import csv
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ENCODINGS_FILE = "face_encodings.csv"

# Load existing face encodings and names from the CSV file
known_face_encodings = []
known_face_names = []

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            encoding = list(map(float, row[1:]))
            known_face_names.append(name)
            known_face_encodings.append(encoding)
    print(f"Loaded {len(known_face_names)} known faces.")
else:
    print("No known faces found. Starting fresh.")

video_capture = cv2.VideoCapture(0)

print("Press 's' to save a new face, 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(frame, "Press 's' to save", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        if name != "Unknown":
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        name = input("Enter the name of the person: ")
        if name:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

            # Save the face data to the CSV file
            with open(ENCODINGS_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([name] + list(face_encodings[0]))

            print(f"Saved face for {name}.")

video_capture.release()
cv2.destroyAllWindows()