import cv2

# Inisialisasi face classifier menggunakan haarcascade default yang sudah tersedia di OpenCV
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load gambar dari path yang sudah ditentukan
image_path = "H:/kuliah/Semester4/prak pcd/GO TO GITHUB/PCDPRIDE/PRAKTEK I5-HAAR CASCADE-FACE AND EYE DETECTION/aep.jpg"
image = cv2.imread(image_path)  # Membaca gambar ke variabel 'image'

# Validasi apakah gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan. Cek path atau nama file.")  # Jika file tidak ada, beri peringatan dan hentikan program
    exit()

# Konversi gambar ke grayscale karena haarcascade bekerja dengan citra grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Deteksi wajah pada gambar grayscale
# Parameter 1.3 = scaleFactor, 5 = minNeighbors (sensitivitas deteksi)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Cek apakah wajah ditemukan atau tidak
if len(faces) == 0:
    print("Tidak ada wajah ditemukan.")  # Informasi jika tidak ada wajah terdeteksi
else:
    # Jika ada wajah, gambarkan kotak persegi panjang di sekeliling wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)  # Warna kotak ungu, ketebalan 2 px

    # Tampilkan gambar dengan wajah yang sudah diberi kotak
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)  # Tunggu input tombol untuk menutup jendela
    cv2.destroyAllWindows()  # Tutup semua jendela OpenCV
