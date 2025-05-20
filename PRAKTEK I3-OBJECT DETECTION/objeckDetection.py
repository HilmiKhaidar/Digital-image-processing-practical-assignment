import cv2
import numpy as np

# Memuat classifier Haar Cascade untuk deteksi mobil
car_cascade = cv2.CascadeClassifier(
    r"H:\kuliah\Semester4\prak pcd\GO TO GITHUB\PCDPRIDE\PRAKTEK I3-OBJECT DETECTION\cars.xml")

# Membuka file video yang akan dianalisis
cam = cv2.VideoCapture(
    r"H:\kuliah\Semester4\prak pcd\GO TO GITHUB\PCDPRIDE\PRAKTEK I3-OBJECT DETECTION\cars.mp4")

# Validasi apakah file classifier berhasil dimuat
if car_cascade.empty():
    print("Gagal membuka file cars.xml")
    exit()

# Validasi apakah file video berhasil dibuka
if not cam.isOpened():
    print("Gagal membuka file cars.mp4")
    exit()

# Loop utama untuk membaca frame video satu per satu
while True:
    ret, frame = cam.read()  # Membaca frame dari video

    # Jika frame tidak terbaca (misalnya video sudah selesai), keluar dari loop
    if not ret:
        break

    # Mengubah frame ke grayscale agar deteksi lebih efisien
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi objek mobil dalam frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    # Memberi kotak hijau di sekitar mobil yang terdeteksi
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Menampilkan hasil deteksi dalam jendela
    cv2.imshow('video', frame)

    # Tekan tombol 'q' untuk keluar dari tampilan
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Melepaskan resource video dan menutup semua jendela OpenCV
cam.release()
cv2.destroyAllWindows()
