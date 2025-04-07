import cv2
import time
import csv

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = 'video1.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Dictionary to track face detection durations
face_timers = {}
permanent_faces = []

# Open a CSV file to log detected faces
with open('detected_faces.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'X', 'Y', 'Width', 'Height', 'Permanent'])

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()
        new_face_timers = {}

        for (x, y, w, h) in faces:
            face_key = (x, y, w, h)
            if face_key in face_timers:
                # Update the timer if the face is already being tracked
                if current_time - face_timers[face_key] >= 2:
                    if face_key not in permanent_faces:
                        permanent_faces.append(face_key)
                        # Log the permanent face to the CSV file
                        csv_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), x, y, w, h, 'Yes'])
                else:
                    new_face_timers[face_key] = face_timers[face_key]
            else:
                # Start tracking a new face
                new_face_timers[face_key] = current_time
                # Log the new face to the CSV file
                csv_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), x, y, w, h, 'No'])

            # Draw a green rectangle for detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Update the face timers
        face_timers = new_face_timers

        # Draw permanent red rectangles for faces detected for over 2 seconds
        for (x, y, w, h) in permanent_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('Face Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()