# Object Detection for Employee Attendance via CCTV Footage 

## I have the CNN model instead of HOG model for Face Recognition to enhance accuracy when multiple faces are used

# Steps that I used
1. First using the `recognize_face_webcam.py` file we will detect, recognize and save the face data in the ``face_encodings.pkl``.
2. Then using the previously generated ``face_encodings.pkl`` and an input ``video_file.mp4`` we will traverse the video and recognize the faces in the video using the `recognize_face_video.py` file.
3. Finally when we will finish the video a part of the  `recognize_face_video.py` file will help us save the **first** and the **last** occurences of the faces in the videos which will help us generate the **In_Time** of the employee and the **Out_Time** of the employee.
4. The ``recognized_faces_log.csv`` file will contain the *Name, First Seen, Last Seen* entries of the employee.
5. The ``recognize_face_live.py`` file will detect and recognize faces that are already saved with ``face_encodings.pkl`` and then saves the **In_Time** of the employee and the **Out_Time** of the employee in the ``recognized_faces_live_log.csv`` file.

# Future Enhacenments