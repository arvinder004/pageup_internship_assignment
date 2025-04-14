import cv2
import pickle
import os
import face_recognition

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter Your Name: ")

data_file = 'data/face_data.pkl'
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        face_data = pickle.load(f)
else:
    face_data = {}

if name not in face_data:
    face_data[name] = []

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        if w < 50 or h < 50:
            continue

        crop_img = frame[y:y+h, x:x+w, :]
        rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)

        if len(encodings) > 0:
            face_data[name].append(encodings[0])

        cv2.putText(frame, f"Collected: {len(face_data[name])}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(face_data[name]) >= 200:
        break

video.release()
cv2.destroyAllWindows()

with open(data_file, 'wb') as f:
    pickle.dump(face_data, f)

print(f"Data for {name} has been saved successfully.")