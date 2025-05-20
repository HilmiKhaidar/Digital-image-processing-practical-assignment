import cv2
import numpy as np

# Gunakan path absolut (disarankan jika file sering tidak terbaca)
img_path = r'h:/kuliah/Semester4/prak pcd/GO TO GITHUB/PCDPRIDE/PRAKTEK I7-CIRCLE HOUGH TRANSFORM/u.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Cek apakah gambar berhasil dibaca
if img is None:
    print(f"Gagal membuka gambar! Pastikan file '{img_path}' ada dan bisa dibaca.")
    exit()

# Proses Hough Circle Transform
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Gambar lingkaran
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Gambar pusat lingkaran
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Hough Circle Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
