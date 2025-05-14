import sys  # Untuk mengakses argumen dan keluar dari aplikasi
import cv2  # OpenCV untuk pemrosesan gambar
import numpy as np  # NumPy untuk operasi numerik
import matplotlib.pyplot as plt  # Untuk menampilkan histogram
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog  # Komponen GUI
from PyQt5.QtGui import QImage, QPixmap  # Untuk konversi dan tampilan gambar
from PyQt5 import QtCore  # Fitur tambahan dari PyQt5
from PyQt5.uic import loadUi  # Untuk memuat file UI yang dibuat dengan Qt Designer


class ShowImage(QMainWindow):  # Kelas utama yang mengatur antarmuka
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)  # Memuat desain GUI dari file .ui
        self.image = None  # Menyimpan gambar asli
        self.processed_image = None  # Menyimpan hasil pemrosesan gambar

        # Menghubungkan tombol dan menu ke fungsi yang sesuai
        self.LoadButton.clicked.connect(self.loadClicked)
        self.actionGrayscale.triggered.connect(self.grayClicked)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.simpleContrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionTranslation.triggered.connect(self.translation)
        self.actionRotation.triggered.connect(self.rotation)
        self.actionResize.triggered.connect(self.resizeImage)
        self.actionCrop.triggered.connect(self.cropImage)
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)

    def loadClicked(self):
        """ Fungsi untuk memuat gambar melalui dialog file """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.loadImage(file_name)

    def loadImage(self, fname):
        """ Memuat gambar dan menyimpannya ke variabel """
        print(f"Loading image: {fname}")  # Menampilkan nama file
        self.image = cv2.imread(fname)  # Membaca gambar

        if self.image is None:
            print("Error: Gambar tidak ditemukan!")
            return

        print(f"Image shape: {self.image.shape}")  # Menampilkan dimensi gambar
        self.processed_image = self.image.copy()  # Simpan salinan gambar asli
        self.displayImage()  # Tampilkan gambar

    def displayImage(self, processed=False):
        """ Menampilkan gambar asli atau hasil pemrosesan ke GUI """
        img_to_show = self.processed_image if processed else self.image
        if img_to_show is None:
            print("Error: Tidak ada gambar yang dimuat!")
            return

        print(f"Displaying image with shape: {img_to_show.shape}")  # Debugging dimensi
        qformat = QImage.Format_RGB888
        if len(img_to_show.shape) == 2:
            qformat = QImage.Format_Grayscale8
        else:
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)

        img = QImage(
            img_to_show.data,
            img_to_show.shape[1],
            img_to_show.shape[0],
            img_to_show.strides[0],
            qformat,
        )

        if processed:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setScaledContents(True)
        else:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setScaledContents(True)

    def applyProcessing(self, processed_img):
        """ Menyimpan dan menampilkan gambar hasil pemrosesan """
        if processed_img is not None:
            print("Applying processing...")  # Debug info
            self.processed_image = processed_img
            self.displayImage(processed=True)

    # ===== OPERASI TITIK =====
    def grayClicked(self):
        """ Mengubah gambar menjadi grayscale manual """
        print("Grayscale clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        self.applyProcessing(gray_img)

    def negative(self):
        """ Mengubah gambar ke negatif """
        print("Negative clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        neg_img = 255 - self.image
        self.applyProcessing(neg_img)

    def biner(self):
        """ Konversi gambar ke biner (hitam putih) """
        print("Biner clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        threshold = 128
        bin_img = np.where(gray_img > threshold, 255, 0).astype(np.uint8)
        self.applyProcessing(bin_img)

    def brightness(self):
        """ Menambahkan brightness ke gambar """
        print("Brightness clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        beta = 50
        bright_img = np.clip(self.image.astype(np.int16) + beta, 0, 255).astype(np.uint8)
        self.applyProcessing(bright_img)

    def simpleContrast(self):
        """ Meningkatkan kontras sederhana """
        print("Simple Contrast clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        alpha = 2.0
        contrast_img = np.clip(self.image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        self.applyProcessing(contrast_img)

    def contrastStretching(self):
        """ Normalisasi kontras menggunakan min-max """
        print("Contrast Stretching clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        min_val, max_val = np.min(gray_img), np.max(gray_img)
        stretched_img = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.applyProcessing(stretched_img)

    # ===== OPERASI HISTOGRAM =====
    def grayHistogram(self):
        """ Menampilkan histogram grayscale """
        print("Grayscale Histogram clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        self.applyProcessing(gray_img)
        plt.hist(gray_img.ravel(), 256, [0, 256], color="gray")
        plt.title("Histogram Grayscale")
        plt.show()

    def RGBHistogram(self):
        """ Menampilkan histogram RGB """
        print("RGB Histogram clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
        plt.title("Histogram RGB")
        plt.show()

    def EqualHistogram(self):
        """ Melakukan histogram equalization """
        print("Histogram Equalization clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        hist, _ = np.histogram(gray_img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        eq_img = np.interp(gray_img.flatten(), range(256), cdf_normalized).reshape(gray_img.shape)
        eq_img = (eq_img * 255 / eq_img.max()).astype(np.uint8)
        self.applyProcessing(eq_img)

        plt.hist(eq_img.ravel(), 256, [0, 256], color="r")
        plt.title("Histogram Equalization")
        plt.show()

    # ===== OPERASI GEOMETRI =====
    def translation(self):
        """ Mentranslasikan gambar """
        print("Translation clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, 50], [0, 1, 50]])
        translated_img = cv2.warpAffine(self.image, M, (cols, rows))
        self.applyProcessing(translated_img)

    def rotation(self):
        """ Merotasi gambar sebesar 45 derajat """
        print("Rotation clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        rotated_img = cv2.warpAffine(self.image, M, (cols, rows))
        self.applyProcessing(rotated_img)

    def resizeImage(self):
        """ Mengubah ukuran gambar menjadi setengahnya """
        print("Resize clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        resized_img = cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.applyProcessing(resized_img)

    def cropImage(self):
        """ Memotong bagian tengah gambar """
        print("Crop clicked!")  # Info debug
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        if rows < 100 or cols < 100:
            print("Error: Gambar terlalu kecil untuk dipotong!")
            return

        cropped_img = self.image[50:rows - 50, 50:cols - 50]
        self.applyProcessing(cropped_img)

    # ===== OPERASI ARITMATIKA DAN LOGIKA =====
    def aritmatika(self):
        """ Penjumlahan dan pengurangan dua gambar grayscale """
        image1 = cv2.imread('Biru.jpg', 0)
        image2 = cv2.imread('Ungu.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey

    def operasiAND(self):
        """ Operasi logika AND pada dua gambar RGB """
        image1 = cv2.imread('Biru.jpg', 1)
        image2 = cv2.imread('Ungu.jpg', 1)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        operasi = cv2.bitwise_and(image1, image2)

        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Operasi AND', operasi)

        cv2.waitKey()

    def operasiOR(self):
        """ Operasi logika OR pada dua gambar RGB """
        image1 = cv2.imread('Biru.jpg', 1)
        image2 = cv2.imread('Ungu.jpg', 1)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        operasi = cv2.bitwise_or(image1, image2)

        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Operasi OR', operasi)

        cv2.waitKey()

    def operasiXOR(self):
        """ Operasi logika XOR pada dua gambar RGB """
        image1 = cv2.imread('Biru.jpg', 1)
        image2 = cv2.imread('Ungu.jpg', 1)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        operasi = cv2.bitwise_xor(image1, image2)

        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Operasi XOR', operasi)

        cv2.waitKey()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle("Image Processing")  # Judul jendela utama
    window.show()
    sys.exit(app.exec_())  # Menjalankan event loop aplikasi
