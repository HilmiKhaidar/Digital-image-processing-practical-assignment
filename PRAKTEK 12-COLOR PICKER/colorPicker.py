import cv2
import numpy as np

def nothing(x):
    pass

# Inisialisasi kamera
cam = cv2.VideoCapture(0)

# Buat jendela untuk trackbar
cv2.namedWindow("Trackbar")

# Buat trackbar untuk HSV
cv2.createTrackbar("L - H", "Trackbar", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbar", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbar", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbar", 255, 255, nothing)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ambil nilai dari trackbar
    l_h = cv2.getTrackbarPos("L - H", "Trackbar")
    l_s = cv2.getTrackbarPos("L - S", "Trackbar")
    l_v = cv2.getTrackbarPos("L - V", "Trackbar")
    u_h = cv2.getTrackbarPos("U - H", "Trackbar")
    u_s = cv2.getTrackbarPos("U - S", "Trackbar")
    u_v = cv2.getTrackbarPos("U - V", "Trackbar")

    # Bentuk array batas bawah dan atas
    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

    # Buat mask dan hasil filter
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Tampilkan hasil
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == 27:  # tekan ESC untuk keluar
        break

cam.release()
cv2.destroyAllWindows()
