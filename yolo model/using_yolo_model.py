import cv2
import face_recognition
import pickle
import csv
from datetime import datetime
from ultralytics import YOLO
import os

ENCODINGS_FILE = "face_encodings_yolo.pkl"
LOG_FILE = "recognized_faces_yolo_log.csv"

model = YOLO("yolov8n.pt")

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
print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    results = model(frame)
    detections = results[0].boxes.xyxy  
    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])  
        face_frame = frame[y1:y2, x1:x2] 

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()