import cv2
import face_recognition
import os
import pickle
from datetime import datetime

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# File to store face encodings
ENCODINGS_FILE = "face_encodings.pkl"

# Load existing face encodings if available
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
else:
    known_face_encodings = []
    known_face_names = []

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Press 's' to save a new face, 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the matched face
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            # Draw a rectangle around the face
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the name of the recognized face
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the video feed
    cv2.imshow("Face Recognition", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # Save the new face
        name = input("Enter the name of the person: ")
        if name:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

            # Save the face encoding to the file
            with open(ENCODINGS_FILE, "wb") as file:
                pickle.dump((known_face_encodings, known_face_names), file)

            print(f"Saved face for {name}.")

# Release resources
video_capture.release()
cv2.destroyAllWindows()