import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ENCODINGS_FILE = "face_encodings.pkl"
LOG_FILE = "recognized_faces_log.csv"

# Load existing face encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Loaded {len(known_face_names)} known faces.")
else:
    print("No known faces found. Please save faces first.")
    exit()

# Dictionary to track first and last occurrences of each face
face_occurrences = {}

# Create or open the log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "First Seen", "Last Seen"])  # Add headers

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

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            # Get the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update first and last occurrences in the dictionary
            if name not in face_occurrences:
                face_occurrences[name] = {"first_seen": current_time, "last_seen": current_time}
            else:
                face_occurrences[name]["last_seen"] = current_time

        # Scale face location back to the original frame size
        top, right, bottom, left = [v * 2 for v in face_location]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display the name of the recognized face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the video frame
    cv2.imshow('Video', frame)

    # Exit the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Write the first and last occurrences to the log file
with open(LOG_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "First Seen", "Last Seen"])  # Add headers
    for name, occurrences in face_occurrences.items():
        writer.writerow([name, occurrences["first_seen"], occurrences["last_seen"]])

# Release resources
video_capture.release()
cv2.destroyAllWindows()
