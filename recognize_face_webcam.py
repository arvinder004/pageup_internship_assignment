# uses the webcam to recognize faces and add names to new faces then save it in .pkl file

import cv2
import face_recognition
import os
import pickle
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ENCODINGS_FILE = "face_encodings.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
else:
    known_face_encodings = []
    known_face_names = []

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

    face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        elif name == "Unknown":
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(frame, "Press 's' to save", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
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
            with open(ENCODINGS_FILE, "wb") as file:
                pickle.dump((known_face_encodings, known_face_names), file)

            print(f"Saved face for {name}.")

video_capture.release()
cv2.destroyAllWindows()