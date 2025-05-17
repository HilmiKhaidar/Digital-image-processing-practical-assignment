import cv2
import numpy as np

# Pakai path absolut
video_path = r"h:\kuliah\Semester4\prak pcd\GO TO GITHUB\A1-F2\PRAKTEK I3-OBJECT DETECTION\cars.avi"
cascade_path = r"h:\kuliah\Semester4\prak pcd\GO TO GITHUB\A1-F2\PRAKTEK I3-OBJECT DETECTION\cars.xml"

camera = cv2.VideoCapture(video_path)
car_cascade = cv2.CascadeClassifier(cascade_path)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(33) == 27:  # Tekan ESC
        break

camera.release()
cv2.destroyAllWindows()
