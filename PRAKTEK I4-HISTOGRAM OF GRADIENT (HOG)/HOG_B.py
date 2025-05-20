import cv2
import imutils

# Inisialisasi HOG descriptor untuk mendeteksi manusia
hog = cv2.HOGDescriptor()

# Set detektor default manusia menggunakan SVM pre-trained
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Baca gambar dari file 'rapat.jpg'
img = cv2.imread("rapat.jpg")

# Resize gambar agar lebar maksimal 400 pixel untuk mempercepat proses
# Jika lebar asli gambar lebih kecil dari 400, maka tidak diubah ukurannya
img = imutils.resize(img, width=min(400, img.shape[1]))

# Deteksi manusia di gambar dengan metode sliding window dan HOG
# winStride = langkah geser window, padding = penambahan margin di sekitar window
# scale = faktor skala untuk image pyramid (multiple ukuran gambar untuk deteksi)
regions, _ = hog.detectMultiScale(img,
                                  winStride=(4, 4),
                                  padding=(4, 4),
                                  scale=1.05)

# Gambarkan kotak berwarna merah di sekitar area manusia yang terdeteksi
for (x, y, w, h) in regions:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Tampilkan gambar hasil deteksi manusia
cv2.imshow("Deteksi Manusia", img)

# Tunggu sampai ada tombol ditekan, kemudian tutup semua jendela OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()
