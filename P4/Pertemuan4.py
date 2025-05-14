#Import Library
import sys
import cv2
import numpy as np
import math
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import math
import tkinter as tk #gui
from tkinter import filedialog #gui

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.pushButton.clicked.connect(self.fungsi)
        self.pushButton2.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contraststretching)
        self.actionNegative_Image.triggered.connect(self.negativeimage)
        self.actionBiner_Image.triggered.connect(self.binerimage)
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionEqualization.triggered.connect(self.EqualizationHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action60_Derajat.triggered.connect(self.rotasi60derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.action2x.triggered.connect(self.zoom2x)
        self.action4x.triggered.connect(self.zoom4x)
        self.action6x.triggered.connect(self.zoom6x)
        self.action8x.triggered.connect(self.zoom8x)
        self.action10x.triggered.connect(self.zoom10x)
        self.action1_2.triggered.connect(self.reduce2x)
        self.action1_4.triggered.connect(self.reduce4x)
        self.action1_8.triggered.connect(self.reduce8x)
        self.action1_10.triggered.connect(self.reduce10x)
        self.actionCrop.triggered.connect(self.crop)
        self.actionOperasi_AND.triggered.connect(self.opand)
        self.actionOperasi_Or.triggered.connect(self.opor)
        self.actionOperasi_XOR.triggered.connect(self.opxor)
        self.actionPertambahan.triggered.connect(self.pertambahan)
        self.actionPengurangan.triggered.connect(self.pengurangan)
        self.actionPerkalian.triggered.connect(self.perkalian)
        self.actionPembagian.triggered.connect(self.pembagian)
        self.horizontalSlider.valueChanged[int].connect(self.brightness)
        self.horizontalSlider_2.valueChanged[int].connect(self.contrast)
        self.actionKernel_3.triggered.connect(self.kernel1)
        self.actionKernel_4.triggered.connect(self.kernel2)
        self.actionMean.triggered.connect(self.mean)
        self.actionGaussian.triggered.connect(self.gaussian)
        self.actionLaplace.triggered.connect(self.sharpmain)
        self.actionI.triggered.connect(self.sharp1x)
        self.actionII.triggered.connect(self.sharp2x)
        self.actionIII.triggered.connect(self.sharp3x)
        self.actionIV.triggered.connect(self.sharp4x)
        self.actionV.triggered.connect(self.sharp5x)
        self.actionVI.triggered.connect(self.sharp6x)
        self.actionMedian_Filtering.triggered.connect(self.median)
        self.actionMax_Filtering.triggered.connect(self.maxfill)
        self.actionMin_Filtering.triggered.connect(self.minfill)
        self.actionLow_Pass_Filter_2.triggered.connect(self.fourierlpfsmooth)
        self.actionHigh_Pass_Filter_2.triggered.connect(self.fourierhpfedge)
        self.actionSobel_2.triggered.connect(self.sobel)
        self.actionPrewitt_2.triggered.connect(self.prewitt)
        self.actionRoberts.triggered.connect(self.robert)
        self.actionCanny_Edge_2.triggered.connect(self.cannyedge)


    def fungsi(self):
        self.image = cv2.imread('ultramen.jpg')

        self.displayImage(1)

    def grayscale(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] + 0.587
                                     * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def brightness(self):
        #agar menghindari crash
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.image.itemset((i, j), b)


        self.displayImage(1)


    def contrast(self):
        #agar menghindari crash
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a + contrast, 0, 255)

                self.image.itemset((i, j), b)


        self.displayImage(1)

    def contraststretching(self):
        # agar menghindari crash
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.image.shape[:2]
        minV = np.min(self.image)
        maxV = np.max(self.image)
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.image.itemset((i, j), b)

        self.displayImage(1)

    def negativeimage(self):
            # agar menghindari crash
            try:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            contrast = 250
            H, W = self.image.shape[:2]
            for i in range(H):
                for j in range(W):
                    a = self.image.item(i, j)
                    b = np.clip(contrast - a, 0, 255)

                    self.image.itemset((i, j), b)

            self.displayImage(1)

    def grayHistogram(self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    # mengubah citra ke greyscale
                    # f(x,y) = 0.299R + 0.587G + 0.114B
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[1, j, 2], 0, 255)
            self.Image = gray

        except:
            self.Image = cv2.imread('ultramen.jpg', cv2.IMREAD_GRAYSCALE)

        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

        self.displayImage(2)


    def rgbHistogram(self):
        self.image = cv2.imread('ultramen.jpg')
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
            plt.show()
        self.displayImage(2)

    def EqualizationHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color="b")
        plt.hist(self.image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("cdf", "histogram"), loc="upper left")
        plt.show()

    def translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h/4, w/4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.image, T, (w, h))

        self.image = img
        self.displayImage(2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi60derajat(self):
        self.rotasi(60)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasi180derajat(self):
        self.rotasi(180)


    def rotasi(self, degree):
        h,w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w/2,h/2),degree, .7)

        cos = np.abs(rotationMatrix[0,0])
        sin = np.abs(rotationMatrix[0,1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image =cv2.warpAffine(self.image, rotationMatrix,(h, w))

        self.image = rot_image
        self.displayImage(2)

    def zoom2x(self):
        self.zoomIn(2)

    def zoom4x(self):
        self.zoomIn(4)

    def zoom6x(self):
        self.zoomIn(6)

    def zoom8x(self):
        self.zoomIn(8)

    def zoom10x(self):
        self.zoomIn(10)

    def zoomIn(self ,skala):
        resize_img = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Original", self.image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey(0)

    def reduce2x(self):
        self.zoomOut(0.5)

    def reduce4x(self):
        self.zoomOut(0.25)

    def reduce8x(self):
        self.zoomOut(0.13)

    def reduce10x(self):
        self.zoomOut(0.10)

    def zoomOut(self,skala):
        resize_img = cv2.resize(self.image, None, fx=skala, fy= skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Original", self.image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey(0)

    def crop(self):
        row_start = 100
        row_end = 300
        col_start = 100
        col_end = 500
        img = self.image[row_start:row_end, col_start:col_end]
        cv2.imshow("Original", self.image)
        cv2.imshow("Crop", img)
        cv2.waitKey(0)

    def aritmatika(self):
        self.Image1 = cv2.imread('ultramen.jpg', 0)
        self.Image2 = cv2.imread('ultramen2.jpg', 0)
        H, W = self.Image1.shape[:2]
        img = cv2.cvtColor(self.open('ultramen.jpg'), cv2.COLOR_BGR2GRAY)
        self.Image2 = cv2.resize(img, (W, H))

    def pertambahan(self):
        Image1 = cv2.imread('ultramen.jpg', 0)
        Image2 = cv2.imread('ultramen2.jpg', 0)
        hasil = Image1 + Image2
        cv2.imshow("hasil pertambahan", hasil)
        cv2.waitKey()

    def pengurangan(self):
        Image1 = cv2.imread('ultramen.jpg', 0)
        Image2 = cv2.imread('ultramen2.jpg', 0)
        hasil = Image1 - Image2
        cv2.imshow("hasil pengurangan", hasil)
        cv2.waitKey()

    def perkalian(self):
        Image1 = cv2.imread('ultramen.jpg', 0)
        Image2 = cv2.imread('ultramen2.jpg', 0)
        hasil = Image1 * Image2
        cv2.imshow("hasil perkalian", hasil)
        cv2.waitKey()

    def pembagian(self):
        Image1 = cv2.imread('ultramen.jpg', 0)
        Image2 = cv2.imread('ultramen2.jpg', 0)
        hasil = Image1 / Image2
        cv2.imshow("hasil perbagian", hasil)
        cv2.waitKey()

    def boolean(self):
        Image1 = cv2.imread('ultramen.jpg', 1)
        Image2 = cv2.imread('ultramen2.jpg', 1)
        self.Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        H, W = self.Image1.shape[:2]
        img = cv2.cvtColor(self.open(), cv2.COLOR_BGR2RGB)
        self.Image2 = cv2.resize(img, (W, H))

    def opand(self):
        img1 = cv2.imread('ultramen.jpg',  1)
        img2 = cv2.imread('ultramen2.jpg', 1)
        img1 =cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 =cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and =cv2.bitwise_and(img1, img2)
        cv2.imshow('Image 1 Original ', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi AND', op_and)
        cv2.waitKey(0)

    def opor(self):
        img1 = cv2.imread('ultramen.jpg', 1)
        img2 = cv2.imread('ultramen2.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_or = cv2.bitwise_or(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi OR', op_or)
        cv2.waitKey(0)



    def opxor(self):
        img1 = cv2.imread('ultramen.jpg', 1)
        img2 = cv2.imread('ultramen2.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_xor = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Operasi XOR',op_xor)
        cv2.waitKey(0)




    def binerimage(self):
            # agar menghindari crash
            try:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            H, W = self.image.shape[:2]


            for i in range(H):
                for j in range(W):
                    a = self.image.item(i, j)


                    if a == 180:
                        b = 0
                    if a < 180:
                        b = 1
                    if a > 180:
                        b = 255

                    self.image.itemset((i, j), b)


            self.displayImage(1)


    def conv(self, X, F):   #x citra yg dikonvolusi, F kernel konvolusi
        X_height = X.shape[0]   #ukuran tinggi citra
        X_width = X.shape[1]    #ukuran lebar citra
        F_height = F.shape[0]   #ukuran tinggi kernel
        F_width = F.shape[1]    #ukuran lebar kernel
        H = (F_height) // 2     #titik tengah dibagi 2 dibulatkan kebawah
        W = (F_width) // 2      #titik tengah dibagi 2 dibulatkan kebawah
        out = np.zeros((X_height, X_width))
        for i in np.arange(H + 1, X_height - H):    #i kebawah, j ke samping
            for j in np.arange(W + 1, X_width - W): #zona, untuk pergerakan kernel
                sum = 0 #nampung total hasil pixel dan bobot kernel dikali dan ditambah
                for k in np.arange(-H, H + 1):  #pergerakan pixel
                    for l in np.arange(-W, W + 1):#Zona pixel, p1,p2,dst
                        a = X[i + k, j + l]         #pixel didalam citra
                        w = F[H + k, W + l]         #bobot di kernel
                        sum += (w * a)              #perhitungan dengan sum yang telah ada
                out[i, j] = sum         #diposisikan di i,j atau titik tengah
        return out  #citra hasil keluaran

    def kernel1(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],     #kernel filtering1 memperjelas gambar
             [1, 1, 1]]
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def kernel2(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[6, 0, -6],
             [6, 1, -6],    #kernel filtering2 memperjelas gambar
             [6, 0, -6]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def mean(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 9) * np.array(
            [[1, 1, 1],
             [1, 1, 1],       #kernel mean pelembutan menggganti intensitas pixel tsb dngn nilai pixel tetangganya
             [1, 1, 1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def gaussian(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],    #gaussian pengaburan
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 2]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpmain(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0,-1, -2, -1,0],
             [-1,-2,16,-2,-1],      #Sharp memperjelas atau penajaman tepi
             [0, -1, -2,-1,0],
             [0, 0, -1, 0, 0]])

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()


    def sharp1x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0/ 16) * np.array(
            [[-1, -1, -1],
             [-1,  8, -1],       #Sharp memperjelas tepi1x
             [-1, -1, -1]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharp2x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[-1, -1, -1],
             [-1,  9, -1],       #Sharp memperjelas tepi2x
             [-1, -1, -1]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharp3x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, -1, 0],
             [-1, 5, -1],        #Sharp memperjelas gambar3x
             [0, -1, 0]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharp4x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1],
             [-2, 5, -2],        #Sharp memperjelas tepi4x
             [1, -2, 1]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharp5x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1],
             [-2, 4, -2],        #Sharp memperjelas tepi5x
             [1, -2, 1]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharp6x(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, 1, 0],
             [1, -4, 1],         #Sharp memperjelas tepi6x
             [0, 1, 0]]
        )

        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def median(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #citra inputan diubah ke grayscale
        hasil = img.copy()
        h, w = img.shape[:2]    #tinggi dan ukuran baris citra
        for i in np.arange(3, h - 3):   #looping mengecek pixel
            for j in np.arange(3, w - 3):
                neighbors = []  #array kosong menampung pixel
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):          #peningkatan kualitas citra
                        a = img.item(i + k, j + l)  #menampung hasil bacaan filter
                        neighbors.append(a) #menambahkan array
                neighbors.sort()    #mengurutkan neigbors
                median = neighbors[24]  #posisi median
                b = median
                hasil.itemset((i, j), b) #menentukan posisi median dari nilai sebelumnya
        plt.imshow(hasil, cmap = 'gray', interpolation= 'bicubic')
        plt.show()

    def maxfill(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h , w = img.shape[:2]
        for i in np.arange(3, h - 3): #ngecek nilai setiap pixel
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                img.itemset((i,j), b)
        plt.imshow(hasil, cmap = 'gray', interpolation= 'bicubic')
        plt.show()

    def minfill(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                b = min
                hasil.itemset((i, j), b)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.show()

    def fourierlpfsmooth(self): #mereduksi noise
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        Img = np.array([[y[j] * 127 for j in range(256)] for i in
                        range(256)], dtype=np.uint8)

        plt.imshow(Img)


        img = cv2.imread('ultramen.jpg',0)
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT) #proses transofrmasi diskrit #array 2d yang kompleks real dan imaginer
        dft_shift = np.fft.fftshift (dft) #shiftting agar titik pusat ada ditengah arrange

        magnitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))) #perhitungan spektrum dari proses dft shift
                                                #0 real                         #imaginer
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)       #proses center di tengah
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask [mask_area] = 1    #set max dengan nilai 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:,:, 1])) #proses spektrum
        f_ishift=np.fft.ifftshift(fshift) #mengembalikan titik origin ke ujung kiri atas

        img_back= cv2.idft(f_ishift)    #frekuensi ke spasial invers
        img_back= cv2.magnitude(img_back[:,:,0],img_back[:,:,1]) #mengembalikan real dan imaginer

        fig = plt.figure (figsize=(12,12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img, cmap= 'gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(magnitude_spectrum, cmap= 'gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(fshift_mask_mag, cmap= 'gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(img_back, cmap= 'gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def fourierhpfedge(self): #mendeteksi tepi
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += max(y)
        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)
        img = cv2.imread('ultramen.jpg',0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift=np.fft.fftshift(dft)
        magnitude_spectrum= 20 * np.log((cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
        rows,cols = img.shape
        crow,ccol = int(rows/2), int(cols/2)

        mask=np.ones((rows,cols,2),np.uint8)
        r=80
        center=[crow,ccol]
        x,y=np.ogrid[:rows,:cols]
        mask_area=(x-center[0]) ** 2 + (y-center[1]) ** 2 <= r*r #perbedaan dengan tadi cernter circle nya 0 sisanya 1
        mask[mask_area]=0 #diberikan 0, beda dengan tadi

        fshift= dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

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

    def sobel(self):
        img = cv2.imread('ultramen.jpg',cv2.IMREAD_GRAYSCALE) #convert rgb ke grayscale
        sobelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) #kernel sobel sumbu x
        sobely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) #kernel sobel sumbu y
        gx = self.conv(img, sobelx) #konvolusi img kernel sumbu x
        gy = self.conv(img, sobely) #konvolusi img kernel sumbu y
        gradient =np.sqrt(gx ** 2 + gy ** 2) #menghitung  gradien sqrt
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8) #normalisasi panjang gradien dalam range 0-255
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic') #menampilkan output img dalam color mapgray
        plt.show()

    def prewitt(self):
        img = cv2.imread('ultramen.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert rgb ke grayscale
        kernel_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])#kernel sobel sumbu x
        kernel_y = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])#kernel sobel sumbu y
        gx = self.conv(img_gray, kernel_x)
        gy = self.conv(img_gray, kernel_y)
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        gradient_norm = (gradient *255.0 / gradient.max()).astype(np.uint8)
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.show()

    def robert(self):
        img = cv2.imread('ultramen.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([[0, 1], [-1, 0]])
        kernel_y = np.array([[1, 0], [0, -1]])

        # apply sobel kernels to image
        gx = cv2.filter2D(img_gray, -1,  kernel_x)
        gy = cv2.filter2D(img_gray, -1, kernel_y)
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.show()

    def cannyedge(self):
        img = cv2.imread("ultramen.jpg")
        plt.imshow(img[:,:,::-1])
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gauss = (1.0 / 57) * np.array ([[0, 1, 2, 1, 0],
                                       [1, 3, 5, 3, 1],
                                       [2, 5, 9, 5, 2], # langkah1kernel gaussian reduksi noise
                                       [1, 3, 5, 3, 1],
                                       [0, 1, 2, 1, 0]])
        img_out = self.conv(img1,  gauss)
        fig = plt.figure(figsize=(12, 12))

        sobel_x = cv2.Sobel(img_out, cv2.CV_64F, 1, 0, ksize= 3)
        sobel_y = cv2.Sobel(img_out, cv2.CV_64F, 0, 1, ksize= 3)
        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        theta = np.arctan2(sobel_y, sobel_x) #finding gradien, deteksi tepi

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        Z = np.zeros(img1.shape, dtype=np.int32)
        H, W = img1.shape
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle [i, j] <= 180):
                        q = mag[i, j + 1]
                        r = mag[i, j - 1]
                    elif (22.5 <= angle[i, j]< 67.5):
                        q = mag[i + 1, j - 1]
                        r = mag[i - 1, j + 1]
                    elif (67.5 <= angle[i, j]< 112.5):
                        q = mag[i + 1, j]
                        r = mag[i - 1, j]
                    elif (112.5 <= angle[i, j]< 157.5):
                        q = mag[i - 1, j - 1]
                        r = mag[i + 1, j + 1]
                    if (mag[i, j] >= q) and (mag[i, j] >= r):
                        Z[i, j] =mag[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        img_N = Z.astype("uint8")

        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                elif (a > strong):
                    b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        strong = 255
        for i in range (1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if((img_H1[i + j, j - 1] == strong) or
                                (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or
                                (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or
                                (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or
                                (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img_out, cmap='gray')
        ax1.title.set_text('Noise Reduce')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(theta, cmap='gray')
        ax2.title.set_text('Finding Gradien')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img_N, cmap='gray')
        ax3.title.set_text('Non-Maximum Suppression')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_H2, cmap='gray')
        ax4.title.set_text('Hysterisis Thresholding')
        plt.show()


    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # cv membaca image dalam format BGR, PyQt membaca dalam format RGB
        img = img.rgbSwapped()
        # menyimpan gambar hasil load di dalam imgLabel
        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))

            # memposisikan gambar di center
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label2.setPixmap(QPixmap.fromImage(img))
            self.label2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label2.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 4')
window.show()
sys.exit(app.exec_())