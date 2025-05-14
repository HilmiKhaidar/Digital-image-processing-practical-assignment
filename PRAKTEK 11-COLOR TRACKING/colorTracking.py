import cv2
import numpy as np

# Inisialisasi kamera
cam = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cam.read()
    if not ret:
        break

    # Konversi dari BGR ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tentukan range warna biru dalam HSV
    lower_color = np.array([66, 98, 100])
    upper_color = np.array([125, 233, 255])

    # Buat mask untuk mendeteksi warna biru
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Terapkan mask ke frame asli
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Tampilkan hasil
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    # Tekan 'Esc' untuk keluar
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# Bersihkan jendela
cam.release()
cv2.destroyAllWindows()
