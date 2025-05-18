import cv2
import numpy as np

# Pakai path absolut
car_cascade = cv2.CascadeClassifier(r"H:\kuliah\Semester4\prak pcd\GO TO GITHUB\A1-F2\PRAKTEK I3-OBJECT DETECTION\cars.xml")
cam = cv2.VideoCapture(r"H:\kuliah\Semester4\prak pcd\GO TO GITHUB\A1-F2\PRAKTEK I3-OBJECT DETECTION\cars.mp4")

# Cek apakah classifier berhasil dimuat
if car_cascade.empty():
    print("❌ Gagal membuka file cars.xml")
    exit()

# Cek apakah video berhasil dibuka
if not cam.isOpened():
    print("❌ Gagal membuka file cars.mp4")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
