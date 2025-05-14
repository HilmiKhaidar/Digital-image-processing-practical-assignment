import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


def fungsi_konvolusi(X, F):

    image_h, image_w = X.shape
    kernel_h, kernel_w = F.shape
    H, W = kernel_h // 2, kernel_w // 2

    output = np.zeros((image_h, image_w))

    for i in range(H, image_h - H):
        for j in range(W, image_w - W):
            Sum = 0
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    Sum += w * a
            output[i, j] = Sum

    return output


def Median(image):
    h, w = image.shape
    img_out = np.zeros_like(image)

    for i in range(3, h - 3):
        for j in range(3, w - 3):
            neighbors = []
            for k in range(-3, 4):
                for l in range(-3, 4):
                    neighbors.append(image[i + k, j + l])
            neighbors.sort()
            img_out[i, j] = neighbors[24]  # Median dari 7x7 adalah elemen ke-24 (0-based index)

    return img_out


def MaxFil(image):
    h, w = image.shape
    img_out = np.copy(image)

    for i in range(3, h - 3):
        for j in range(3, w - 3):
            max_value = image[i, j]  # Inisialisasi dengan nilai piksel saat ini
            for k in range(-3, 4):
                for l in range(-3, 4):
                    if image[i + k, j + l] > max_value:
                        max_value = image[i + k, j + l]  # Cari nilai maksimum
            img_out[i, j] = max_value  # Set nilai maksimum ke output image

    return img_out


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)  # Load UI dari Qt Designer

        self.image = None  # Variabel untuk menyimpan gambar asli
        self.gray_image = None  # Variabel untuk menyimpan gambar grayscale
        self.binary_image = None  # Variabel untuk menyimpan gambar biner

        # Hubungkan tombol dengan fungsi
        self.Button_LoadCitra.clicked.connect(self.loadClicked)
        self.Button_ProsesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan_2.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBinary.triggered.connect(self.binary)
        self.actionHistogram_Grayscale.triggered.connect(self.grayhistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)

        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90degree.triggered.connect(self.rotasi90derajat)
        self.action_90degree.triggered.connect(self.rotasimin90derajat)
        self.action45degree.triggered.connect(self.rotasi45derajat)
        self.action_45degree.triggered.connect(self.rotasimin45derajat)
        self.action180degree.triggered.connect(self.rotasi180derajat)

        self.action2x.triggered.connect(self.zoomIn2)
        self.action3x.triggered.connect(self.zoomIn3)
        self.action4x.triggered.connect(self.zoomIn4)
        self.action0_5.triggered.connect(self.zoomOutHalf)
        self.action0_25.triggered.connect(self.zoomOutQuarter)
        self.action0_75.triggered.connect(self.zoomOutThreeQuarter)

        self.actionCrop.triggered.connect(self.crop)

        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionBagi_dan_Kali.triggered.connect(self.aritmatika2)

        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)

        self.actionA.triggered.connect(self.filteringclickedA)
        self.actionB.triggered.connect(self.filteringclickedB)
        self.actionA_2.triggered.connect(self.MeanFilterA)
        self.actionB_2.triggered.connect(self.MeanFilterB)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.action1.triggered.connect(self.Sharpening1)
        self.action2.triggered.connect(self.Sharpening2)
        self.action3.triggered.connect(self.Sharpening3)
        self.action4.triggered.connect(self.Sharpening4)
        self.action5.triggered.connect(self.Sharpening5)
        self.action6.triggered.connect(self.Sharpening6)
        self.action7.triggered.connect(self.Sharpening7)
        self.actionMedian_Filter.triggered.connect(self.MedianFilter)
        self.actionMax_Filter.triggered.connect(self.MaxFilter)
        self.actionLow_Pass_Filter.triggered.connect(self.LowPass)
        self.actionHigh_Pass_Filter.triggered.connect(self.HighPass)
        self.actionSobel.triggered.connect(self.Sobel)
        self.actionPrewitt.triggered.connect(self.Prewitt)
        self.actionRoberts.triggered.connect(self.Roberts)
        self.actionCanny.triggered.connect(self.Canny)

    def aritmatika(self):
        image1 = cv2.imread('Biru.jpg', 0)
        image2 = cv2.imread('Ungu.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)

    def aritmatika2(self):
        image1 = cv2.imread('Biru.jpg', 0)
        image2 = cv2.imread('Ungu.jpg', 0)
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_kali)
        cv2.imshow('Image Bagi', image_bagi)

    def operasiAND(self):
        img1 = cv2.imread('Biru.jpg', 1)
        img2 = cv2.imread('Ungu.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and = cv2.bitwise_and(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi AND', op_and)
        cv2.waitKey()

    def operasiOR(self):
        img1 = cv2.imread('Biru.jpg', 1)
        img2 = cv2.imread('Ungu.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_or = cv2.bitwise_or(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi OR', op_or)
        cv2.waitKey()

    def operasiXOR(self):
        img1 = cv2.imread('Biru.jpg', 1)
        img2 = cv2.imread('Ungu.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_xor = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi XOR', op_xor)
        cv2.waitKey()

    def translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.image, T, (w, h))
        self.image = img
        self.displayImage(self.image, self.label_2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasimin90derajat(self):
        self.rotasi(-90)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasimin45derajat(self):
        self.rotasi(-45)

    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (nW, nH))
        self.image = rot_image
        self.displayImage(self.image, self.label_2)

    def zoomIn2(self):
        self.zoom(2)

    def zoomIn3(self):
        self.zoom(3)

    def zoomIn4(self):
        self.zoom(4)

    def zoomOutHalf(self):
        self.zoom(0.5)

    def zoomOutQuarter(self):
        self.zoom(0.25)

    def zoomOutThreeQuarter(self):
        self.zoom(0.75)

    def zoom(self, scale):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if scale > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        resize_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
        cv2.imshow('Original', img)
        cv2.imshow(f'Zoom {"In" if scale > 1 else "Out"} {scale}x', resize_img)
        cv2.waitKey()

    def crop(self):
        self.cropImage(50, 50, 200, 200)

    def cropImage(self, start_row, start_col, end_row, end_col):
        height, width, _ = self.image.shape

        start_row = max(0, min(start_row, height))
        start_col = max(0, min(start_col, width))
        end_row = max(0, min(end_row, height))
        end_col = max(0, min(end_col, width))

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        cropped_img = img[start_row:end_row, start_col:end_col]

        cv2.imshow('Original', img)
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('ejenali.jpg')
        self.displayImage(self.image, self.label)

    def grayscale(self):
        if self.image is None:
            print("Error: Tidak ada gambar untuk dikonversi.")
            return

        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)

        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)

        self.gray_image = gray
        self.displayImage(self.gray_image, self.label_2)

    def filteringclickedA(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def filteringclickedB(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[6, 0, -6],
             [6, 1, -6],
             [6, 0, -6]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def MeanFilterA(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def MeanFilterB(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[1 / 4, 1 / 4],
             [1 / 4, 1 / 4]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Gaussian(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[0.0625, 0.125, 0.0625],
             [0.125, 0.25, 0.125],
             [0.0625, 0.125, 0.0625]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening1(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening2(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening3(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening4(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[1, -2, 1],
             [-2, 5, -2],
             [1, -2, 1]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening5(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening6(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def Sharpening7(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return
        kernel = (1.0 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]]
        )
        img_out = fungsi_konvolusi(self.gray_image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def MedianFilter(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dikonvolusikan.")
            return

        img_out = Median(self.gray_image)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def MaxFilter(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dikonvolusikan.")
            return

        img_out = MaxFil(self.gray_image)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def LowPass(self):
        img = cv2.imread('Kota.jpg', 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def HighPass(self):
        img = cv2.imread('Kota.jpg', 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def Sobel(self):
        img = cv2.imread('ejenali.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        Gx = cv2.filter2D(gray, -1, sobel_x)
        Gy = cv2.filter2D(gray, -1, sobel_y)

        grad = np.sqrt(np.square(Gx) + np.square(Gy))
        grad_normalized = (grad / grad.max()) * 255
        grad_normalized = grad_normalized.astype(np.uint8)

        plt.imshow(grad_normalized, cmap='gray', interpolation='bicubic')
        plt.title('Sobel Edge Detection')
        plt.axis('off')
        plt.show()

    def Prewitt(self):
        img = cv2.imread('ejenali.jpg')
        if img is None:
            raise FileNotFoundError("Gambar tidak ditemukan.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prewitt_x = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float32)

        prewitt_y = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]], dtype=np.float32)

        Gx = cv2.filter2D(gray, -1, prewitt_x)
        Gy = cv2.filter2D(gray, -1, prewitt_y)

        grad = np.sqrt(Gx ** 2 + Gy ** 2)
        grad_normalized = (grad / grad.max()) * 255
        grad_normalized = grad_normalized.astype(np.uint8)

        plt.imshow(grad_normalized, cmap='gray')
        plt.title('Prewitt Edge Detection')
        plt.axis('off')
        plt.show()

    def Roberts(self):
        img = cv2.imread('ejenali.jpg')
        if img is None:
            raise FileNotFoundError("Gambar tidak ditemukan.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        roberts_x = np.array([[1, 0],
                              [0, -1]], dtype=np.float32)

        roberts_y = np.array([[0, 1],
                              [-1, 0]], dtype=np.float32)

        Gx = cv2.filter2D(gray, -1, roberts_x)
        Gy = cv2.filter2D(gray, -1, roberts_y)

        grad = np.sqrt(Gx ** 2 + Gy ** 2)
        grad_normalized = (grad / grad.max()) * 255
        grad_normalized = grad_normalized.astype(np.uint8)

        plt.imshow(grad_normalized, cmap='gray')
        plt.title('Roberts Edge Detection')
        plt.axis('off')
        plt.show()

    def Canny(self):
        img = cv2.imread('ejenali.jpg')
        if img is None:
            raise FileNotFoundError("Gambar tidak ditemukan. Pastikan file 'contoh_gambar.png' tersedia.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gauss = (1.0 / 57) * np.array([
            [0, 1, 2, 1, 0],
            [1, 3, 5, 3, 1],
            [2, 5, 9, 5, 2],
            [1, 3, 5, 3, 1],
            [0, 1, 2, 1, 0]
        ], dtype=np.float32)
        blurred = cv2.filter2D(gray, -1, gauss)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        Gx = cv2.filter2D(blurred, -1, sobel_x)
        Gy = cv2.filter2D(blurred, -1, sobel_y)

        grad = np.sqrt(Gx ** 2 + Gy ** 2)
        grad_normalized = (grad / grad.max()) * 255
        grad_normalized = grad_normalized.astype(np.uint8)
        theta = np.arctan2(Gy, Gx)

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        H, W = grad_normalized.shape
        Z = np.zeros((H, W), dtype=np.float32)
        img_out = grad_normalized.copy()

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img_out[i, j + 1]
                    r = img_out[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img_out[i + 1, j - 1]
                    r = img_out[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img_out[i + 1, j]
                    r = img_out[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img_out[i - 1, j - 1]
                    r = img_out[i + 1, j + 1]
                if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                    Z[i, j] = img_out[i, j]
                else:
                    Z[i, j] = 0
        img_N = Z.astype(np.uint8)

        weak = 100
        strong = 150
        for i in range(H):
            for j in range(W):
                a = img_N[i, j]
                if a > strong:
                    img_N[i, j] = 255
                elif a > weak:
                    img_N[i, j] = weak
                else:
                    img_N[i, j] = 0
        img_H1 = img_N.copy()

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_H1[i, j] == weak:
                    if ((img_H1[i + 1, j - 1] == 255) or (img_H1[i + 1, j] == 255) or (img_H1[i + 1, j + 1] == 255) or
                            (img_H1[i, j - 1] == 255) or (img_H1[i, j + 1] == 255) or
                            (img_H1[i - 1, j - 1] == 255) or (img_H1[i - 1, j] == 255) or (
                                    img_H1[i - 1, j + 1] == 255)):
                        img_H1[i, j] = 255
                    else:
                        img_H1[i, j] = 0
        img_H2 = img_H1.copy()

        titles = [
            "1. Grayscale",
            "2. Gaussian Blurred",
            "3. Gradien Magnitude",
            "4. Non-Max Suppression",
            "5. Hysteresis (Part 1)",
            "6. Final Canny Edge"
        ]
        images = [gray, blurred, grad_normalized, img_N, img_N, img_H2]

        plt.figure(figsize=(15, 10))
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def EqualHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(self.image, self.label_2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def RGBHistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
        plt.plot(histo, color=col)
        plt.xlim([0, 256])
        plt.show()

    def grayhistogram(self):
        if self.image is None:
            print("Error: Tidak ada gambar untuk dikonversi.")
            return

        H, W = self.image.shape[:2]  # Ambil tinggi dan lebar gambar
        gray = np.zeros((H, W), np.uint8)  # Buat array kosong untuk grayscale

        # Looping setiap piksel untuk konversi ke grayscale
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)

        self.gray_image = gray  # Simpan hasil grayscale
        self.displayImage(self.gray_image, self.label_2)  # Tampilkan di label hasil
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    def brightness(self):

        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dicerahkan.")
            return

        H, W = self.gray_image.shape  # Ambil tinggi dan lebar gambar grayscale
        brightness = -50  # Faktor pencerahan

        for i in np.arange(H):
            for j in np.arange(W):
                a = self.gray_image[i, j]  # Ambil nilai piksel
                b = np.clip(a.astype(np.float32) + brightness, 0, 255).astype(np.uint8)  # Lakukan clipping
                self.gray_image[i, j] = b  # Simpan hasil

        # Menampilkan gambar grayscale hasil pencerahan di label tempat RGB awal
        self.displayImage(self.gray_image, self.label)

    def contrast(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dikontraskan.")
            return

        H, W = self.gray_image.shape  # Ambil tinggi dan lebar gambar grayscale
        contrast = 1  # Faktor kontras

        for i in np.arange(H):
            for j in np.arange(W):
                a = self.gray_image[i, j]  # Ambil nilai piksel
                b = np.clip(a * contrast, 0, 255)  # Lakukan clipping
                self.gray_image[i, j] = b  # Simpan hasil

        # Menampilkan gambar grayscale hasil pencerahan di label tempat RGB awal
        self.displayImage(self.gray_image, self.label)

    def contrastStreching(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dikontraskan.")
            return

        H, W = self.gray_image.shape  # Ambil tinggi dan lebar gambar grayscale
        minV = np.min(self.gray_image)
        maxV = np.max(self.gray_image)

        for i in np.arange(H):
            for j in np.arange(W):
                a = self.gray_image[i, j]  # Ambil nilai piksel
                b = float(a - minV) / (maxV - minV) * 255

                self.gray_image[i, j] = b  # Simpan hasil

        # Menampilkan gambar grayscale hasil pencerahan di label tempat RGB awal
        self.displayImage(self.gray_image, self.label)
        cv2.imwrite('A9-C2/contrast1.7.jpg', self.gray_image)  # Simpan sebagai JPG

    def negative(self):
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dinegatifkan.")
            return

        self.gray_image = 255 - self.gray_image

        # Menampilkan gambar grayscale hasil pencerahan di label tempat RGB awal
        self.displayImage(self.gray_image, self.label)

    def binary(self):
        """Mengubah gambar grayscale menjadi citra biner dengan threshold 180."""
        if self.gray_image is None:
            print("Error: Tidak ada gambar grayscale untuk dikonversi ke biner.")
            return

        THRESHOLD = 180  # Batas thresholding

        # Proses thresholding: nilai di atas 180 jadi 255, di bawahnya jadi 0
        binary_image = np.where(self.gray_image > THRESHOLD, 255, 0).astype(np.uint8)

        self.binary_image = binary_image  # Simpan hasil biner
        self.displayImage(self.binary_image, self.label)  # Tampilkan di label awal (RGB)

    def loadImage(self, filename):
        """Membaca gambar dari file dan mengonversinya ke format RGB"""
        self.image = cv2.imread(filename)
        if self.image is None:
            print("Error: Gambar tidak ditemukan atau tidak bisa dibaca.")
            return

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB

    def displayImage(self, img, label):
        """Menampilkan gambar di QLabel yang ditentukan"""
        if img is None:
            print("Error: Tidak ada gambar untuk ditampilkan.")
            return

        # Cek apakah gambar grayscale atau berwarna
        if len(img.shape) == 2:  # Grayscale
            qformat = QImage.Format_Grayscale8
        else:  # RGB
            qformat = QImage.Format_RGB888

        qimg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        # Set gambar ke QLabel yang diberikan sebagai parameter
        label.setPixmap(QPixmap.fromImage(qimg))
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label.setScaledContents(True)  # Agar gambar menyesuaikan QLabel


# Inisialisasi aplikasi
app = QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan_4')
window.show()
sys.exit(app.exec_())