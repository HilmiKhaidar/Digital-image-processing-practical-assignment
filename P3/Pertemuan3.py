import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from PyQt5.uic import loadUi


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)  # Pastikan file GUI.ui ada di folder yang samaaa
        self.image = None
        self.processed_image = None  # Menyimpan hasil pemrosesan sementara

        # Menghubungkan tombol dengan fungsi
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
        self.actionKonvolusi_A.triggered.connect(self.konvolusiA)
        self.actionKonvolusi_B.triggered.connect(self.konvolusiB)
        self.actionKonvolusi_i.triggered.connect(self.konvolusiI)
        self.actionKonvolusi_ii.triggered.connect(self.konvolusiII)
        self.actionGaussian.triggered.connect(self.konvolusiGaussian)
        self.actionSharp_i.triggered.connect(self.konvolusiSharpI)
        self.actionSharp_ii.triggered.connect(self.konvolusiSharpII)
        self.actionSharp_iii.triggered.connect(self.konvolusiSharpIII)
        self.actionSharp_iv.triggered.connect(self.konvolusiSharpIV)
        self.actionSharp_v.triggered.connect(self.konvolusiSharpV)
        self.actionSharp_vi.triggered.connect(self.konvolusiSharpVI)
        self.actionLaplace.triggered.connect(self.konvolusiLaplace)
        self.actionMedian.triggered.connect(self.medianFilterClicked)
        self.actionMax.triggered.connect(self.maxFilterClicked)

    def loadClicked(self):
        """ Memuat gambar dari file """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.loadImage(file_name)

    def loadImage(self, fname):
        """ Memuat gambar dari file """
        print(f"Loading image: {fname}")  # Debugging path
        self.image = cv2.imread(fname)

        if self.image is None:
            print("Error: Gambar tidak ditemukan!")
            return

        print(f"Image shape: {self.image.shape}")  # Debugging shape gambar
        self.processed_image = self.image.copy()  # Simpan salinan asli
        self.displayImage()

    def displayImage(self, processed=False):
        """ Menampilkan gambar di GUI """
        img_to_show = self.processed_image if processed else self.image
        if img_to_show is None:
            print("Error: Tidak ada gambar yang dimuat!")
            return

        print(f"Displaying image with shape: {img_to_show.shape}")  # Debugging shape gambar
        qformat = QImage.Format_RGB888
        if len(img_to_show.shape) == 2:  # Jika grayscale
            qformat = QImage.Format_Grayscale8
        else:
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)  # Konversi ke RGB untuk ditampilkan

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
        """ Menyimpan hasil pemrosesan dan menampilkan di hasilLabel """
        if processed_img is not None:
            print("Applying processing...")  # Debugging
            self.processed_image = processed_img
            self.displayImage(processed=True)

    # Operasi Titik
    def grayClicked(self):
        """ Konversi gambar ke grayscale menggunakan rumus manual """
        print("Grayscale clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Rumus manual untuk grayscale: Y = 0.299*R + 0.587*G + 0.114*B
        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        self.applyProcessing(gray_img)

    def negative(self):
        """ Konversi gambar ke negatif menggunakan rumus manual """
        print("Negative clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Rumus manual untuk negatif: 255 - pixel value
        neg_img = 255 - self.image
        self.applyProcessing(neg_img)

    def biner(self):
        """ Konversi gambar ke biner (hitam putih) menggunakan rumus manual """
        print("Biner clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Konversi ke grayscale terlebih dahulu
        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # Rumus manual untuk biner: thresholding
        threshold = 128
        bin_img = np.where(gray_img > threshold, 255, 0).astype(np.uint8)
        self.applyProcessing(bin_img)

    def brightness(self):
        """ Menambahkan kecerahan menggunakan rumus manual """
        print("Brightness clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Rumus manual untuk brightness: pixel value + beta
        beta = 50  # Nilai default untuk brightness
        bright_img = np.clip(self.image.astype(np.int16) + beta, 0, 255).astype(np.uint8)
        self.applyProcessing(bright_img)

    def simpleContrast(self):
        """ Simple contrast enhancement menggunakan rumus manual """
        print("Simple Contrast clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Rumus manual untuk contrast: pixel value * alpha
        alpha = 2.0  # Nilai default untuk contrast
        contrast_img = np.clip(self.image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        self.applyProcessing(contrast_img)

    def contrastStretching(self):
        """ Kontras stretching dengan min-max normalisasi menggunakan rumus manual """
        print("Contrast Stretching clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Konversi ke grayscale terlebih dahulu
        gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # Rumus manual untuk contrast stretching
        min_val, max_val = np.min(gray_img), np.max(gray_img)
        stretched_img = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.applyProcessing(stretched_img)

    # Operasi Histogram
    def grayHistogram(self):
        """ Menampilkan histogram grayscale """
        print("Grayscale Histogram clicked!")  # Debugging
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
        print("RGB Histogram clicked!")  # Debugging
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
        """ Histogram Equalization pada gambar grayscale """
        print("Histogram Equalization clicked!")  # Debugging
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

    # Operasi Geometri
    def translation(self):
        """ Translasi gambar """
        print("Translation clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, 50], [0, 1, 50]])  # Translasi 50 piksel ke kanan dan bawah
        translated_img = cv2.warpAffine(self.image, M, (cols, rows))
        self.applyProcessing(translated_img)

    def rotation(self):
        """ Rotasi gambar """
        print("Rotation clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotasi 45 derajat
        rotated_img = cv2.warpAffine(self.image, M, (cols, rows))
        self.applyProcessing(rotated_img)

    def resizeImage(self):
        """ Resize gambar """
        print("Resize clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        resized_img = cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # Resize 50%
        self.applyProcessing(resized_img)

    def cropImage(self):
        """ Crop gambar """
        print("Crop clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        rows, cols = self.image.shape[:2]
        if rows < 100 or cols < 100:  # Pastikan gambar cukup besar untuk dipotong
            print("Error: Gambar terlalu kecil untuk dipotong!")
            return

        cropped_img = self.image[50:rows - 50, 50:cols - 50]  # Crop 50 piksel dari setiap sisi
        self.applyProcessing(cropped_img)

    def aritmatika(self):
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
        image1 = cv2.imread('Biru.jpg', 1)
        image2 = cv2.imread('Ungu.jpg', 1)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        operasi = cv2.bitwise_xor(image1, image2)

        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Operasi XOR', operasi)

        cv2.waitKey()

    # ================= PERTEMUAN 3 =============
    def konvolusi_manual(self, input_img, kernel):
        """
        Melakukan konvolusi manual pada citra

        Parameters:
        input_img: Citra input (array numpy)
        kernel: Kernel konvolusi (array numpy)

        Returns:
        output: Citra hasil konvolusi
        """
        print("Melakukan konvolusi manual...")  # Debugging

        # 1. Baca ukuran tinggi dan lebar citra
        height, width = input_img.shape

        # 2. Baca ukuran tinggi dan lebar kernel
        k_height, k_width = kernel.shape

        # 3. H=ukuran tinggi kernel /2
        H = k_height // 2

        # 4. W=ukuran lebar kernel/2
        W = k_width // 2

        # Buat citra output dengan ukuran yang sama
        output = np.zeros_like(input_img)

        # 5. For i: H+1 to ukuran_tinggi_citra-H
        #    For j: W+1 to ukuran_lebar_citra-W
        for i in range(H, height - H):
            for j in range(W, width - W):
                # Lakukan konvolusi
                sum_val = 0
                # For k: -H to H
                for k in range(-H, H + 1):
                    # For l: -W to W
                    for l in range(-W, W + 1):
                        # a=X[i+k, j+l]
                        a = input_img[i + k, j + l]
                        # w=F[H+k, W+l]
                        w = kernel[H + k, W + l]
                        # sum=sum+(w*a)
                        sum_val += w * a

                # out[i, j] = sum
                output[i, j] = np.clip(sum_val, 0, 255)

        return output

    def konvolusi_manual_improved(self, input_img, kernel):
        """
        Melakukan konvolusi manual pada citra (mendukung kernel dimensi genap dan ganjil)

        Parameters:
        input_img: Citra input (array numpy)
        kernel: Kernel konvolusi (array numpy)

        Returns:
        output: Citra hasil konvolusi
        """
        print("Melakukan konvolusi manual...")  # Debugging

        # 1. Baca ukuran tinggi dan lebar citra
        height, width = input_img.shape

        # 2. Baca ukuran tinggi dan lebar kernel
        k_height, k_width = kernel.shape

        # Hitung padding berdasarkan ukuran kernel (berfungsi untuk genap & ganjil)
        pad_h = k_height // 2
        pad_w = k_width // 2

        # Tambahan padding untuk kernel dimensi genap
        pad_h_right = pad_h if k_height % 2 == 1 else pad_h
        pad_w_right = pad_w if k_width % 2 == 1 else pad_w

        # Buat citra output dengan ukuran yang sama
        output = np.zeros_like(input_img)

        # Loop melalui setiap pixel valid di citra
        for i in range(pad_h, height - pad_h_right):
            for j in range(pad_w, width - pad_w_right):
                # Lakukan konvolusi
                sum_val = 0
                # Loop melalui kernel
                for ki in range(k_height):
                    for kj in range(k_width):
                        # Hitung posisi pixel di citra input
                        i_pos = i + (ki - pad_h)
                        j_pos = j + (kj - pad_w)

                        # Pastikan posisi valid
                        if 0 <= i_pos < height and 0 <= j_pos < width:
                            # Ambil nilai pixel dan bobot kernel
                            pixel_val = input_img[i_pos, j_pos]
                            kernel_val = kernel[ki, kj]
                            # Tambahkan ke jumlah
                            sum_val += pixel_val * kernel_val

                # Simpan nilai hasil konvolusi
                output[i, j] = np.clip(sum_val, 0, 255)

        return output

    def median_filter(self, input_img):
        """
        Melakukan median filtering pada citra

        Parameters:
        input_img: Citra input (array numpy)

        Returns:
        output: Citra hasil median filtering
        """
        print("Melakukan median filtering...")  # Debugging

        # 1. Konversi citra ke grayscale
        if len(input_img.shape) == 3:
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = input_img.copy()

        # 2. Copy image untuk hasil output
        img_out = gray_img.copy()

        # 3. Ukuran tinggi dan lebar citra
        h = gray_img.shape[0]
        w = gray_img.shape[1]

        # 4. Loop untuk setiap piksel valid di citra (dengan jendela 3x3)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Inisialisasi list untuk menyimpan nilai neighbor
                neighbors = []

                # Kumpulkan nilai piksel tetangga dalam jendela 3x3
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        # Ambil nilai piksel tetangga
                        a = gray_img[i + k, j + l]
                        # Tambahkan ke list neighbors
                        neighbors.append(a)

                # Mengurutkan neighbors
                neighbors.sort()

                # Mencari nilai median (posisi ke-4 untuk jendela 3x3 setelah diurutkan)
                median = neighbors[4]  # Posisi tengah dari 9 piksel

                # Menempatkan nilai median ke piksel output
                img_out[i, j] = median

        return img_out

    def medianFilterClicked(self):
        """
        Handler untuk tombol Median Filter
        """
        print("Median Filter clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Lakukan median filtering
        img_out = self.median_filter(self.image)

        # Simpan hasil untuk ditampilkan di GUI
        self.applyProcessing(img_out)

    def konvolusiA(self):
        kernel = np.ones((3, 3)) / 9
        self.filteringClicked(kernel)

    def konvolusiB(self):
        kernel = np.array([[6, 0, -6], [6, 1, -6], [6, 0, -6]])
        self.filteringClicked(kernel)

    def konvolusiI(self):
        kernel = np.ones((3, 3)) / 9
        self.filteringClicked(kernel)

    def konvolusiII(self):
        kernel = np.ones((2, 2)) / 4
        self.filteringClicked(kernel)

    def konvolusiGaussian(self):
        kernel = (1.0 / 345) * np.array([
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]])
        self.filteringClicked(kernel)

    def konvolusiII(self):
        kernel = np.ones((2, 2)) / 4
        self.filteringClicked(kernel)

    def konvolusiSharpI(self):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        self.filteringClicked(kernel)

    def konvolusiSharpII(self):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        self.filteringClicked(kernel)

    def konvolusiSharpIII(self):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        self.filteringClicked(kernel)

    def konvolusiSharpIV(self):
        kernel = np.array([
            [1, -2, 1],
            [-2, 5, -2],
            [1, -2, 1]
        ])
        self.filteringClicked(kernel)

    def konvolusiSharpV(self):
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        self.filteringClicked(kernel)

    def konvolusiSharpVI(self):
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        self.filteringClicked(kernel)
    def konvolusiLaplace(self):
        kernel = (1.0 / 16) * np.array([[0, 0, -1, 0, 0],
[0, -1, -2, -1, 0],
[-1, -2, 16, -2, -1],
[0, -1, -2, -1, 0],
[0, 0, -1, 0, 0]])
        self.filteringClicked(kernel)

    def filteringClicked(self, kernel):
        """
        Melakukan filtering pada citra menggunakan konvolusi manual
        """
        print("Filtering clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Ubah citra masukan menjadi grayscale
        if len(self.image.shape) == 3:
            gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray_img = self.image.copy()

        # Uji menggunakan 2 kernel tetapi 1 yang akan digunakan, opsi lain bisa dicomment
        # Kernel 1: Box filter (blur)

        kernel_strong_blur = np.ones((9, 9)) / 81
        # Lakukan konvolusi
        img_out = self.konvolusi_manual_improved(gray_img, kernel)

        # Simpan hasil untuk ditampilkan di GUI
        self.applyProcessing(img_out)

    def mean_filter(self, image, kernel_type='3x3'):
        """
        Melakukan pelembutan citra menggunakan mean filter

        Parameters:
        image: Citra input (array numpy)
        kernel_type: Jenis kernel yang digunakan ('3x3' atau '2x2')

        Returns:
        output: Citra hasil pelembutan
        """
        # Pilih kernel sesuai tipe
        if kernel_type == '3x3':
            # Kernel (i) ukuran 3x3
            kernel = np.ones((3, 3)) / 9
            print(f"kernel type: {kernel_type}")
        elif kernel_type == '2x2':
            # Kernel (ii) ukuran 2x2
            kernel = np.ones((2, 2)) / 4
        else:
            raise ValueError("Kernel type not supported")
        print
        # Lakukan konvolusi menggunakan fungsi konvolusi manual
        filtered_image = self.konvolusi_manual_improved(image, kernel)
        print(f"kernel : {kernel}")

        return filtered_image

    def max_filter(self, input_img):
        """
        Melakukan maximum filtering pada citra

        Parameters:
        input_img: Citra input (array numpy)

        Returns:
        output: Citra hasil maximum filtering
        """
        print("Melakukan maximum filtering...")  # Debugging

        # 1. Konversi citra ke grayscale
        if len(input_img.shape) == 3:
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = input_img.copy()

        # 2. Copy image untuk hasil output
        img_out = gray_img.copy()

        # 3. Ukuran tinggi dan lebar citra
        h = gray_img.shape[0]
        w = gray_img.shape[1]

        # 4. Loop untuk setiap piksel valid di citra
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                # Inisialisasi nilai minimum (akan diubah menjadi nilai maksimum)
                max_val = 0  # Nilai awal untuk max

                # Kumpulkan nilai piksel tetangga dalam jendela yang lebih besar (7x7)
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        # Baca piksel pada posisi (i+k, j+l)
                        a = gray_img[i + k, j + l]

                        # Perbarui nilai maximum jika ditemukan nilai yang lebih besar
                        if a > max_val:
                            max_val = a

                # Tempatkan nilai maksimum ke piksel output
                img_out[i, j] = max_val

        return img_out

    def maxFilterClicked(self):
        """
        Handler untuk tombol Max Filter
        """
        print("Max Filter clicked!")  # Debugging
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Lakukan max filtering
        img_out = self.max_filter(self.image)

        # Simpan hasil untuk ditampilkan di GUI
        self.applyProcessing(img_out)

    def pelembutan_citra(self):
        """
        Menu untuk pelembutan citra
        """
        if self.image is None:
            print("Error: Tidak ada gambar!")
            return

        # Ubah citra masukan menjadi grayscale jika berwarna
        if len(self.image.shape) == 3:
            gray_img = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray_img = self.image.copy()

        # Dialog untuk memilih jenis kernel
        kernel_options = ['3x3 Mean Filter', '2x2 Mean Filter']
        # Di sini seharusnya ada dialog pilihan di GUI
        # Untuk contoh ini, asumsikan user memilih kernel 3x3
        selected_kernel = '2x2'  # atau '2x2' tergantung pilihan

        # Lakukan pelembutan dengan kernel yang dipilih
        result_image = self.mean_filter(gray_img, selected_kernel)

        # Tampilkan hasil
        self.applyProcessing(result_image)

        # Tambahkan analisis pixel sebelum dan sesudah konvolusi
        # Contoh: menampilkan histogram atau nilai statistik
        print(f"Nilai rata-rata sebelum: {np.mean(gray_img)}")
        print(f"Nilai rata-rata sesudah: {np.mean(result_image)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle("Image Processing")
    window.show()
    sys.exit(app.exec_())