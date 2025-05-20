import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.color import rgb2gray

# Load citra warna (RGB) contoh dari library skimage
image = data.astronaut()

# Ubah citra warna ke grayscale untuk pemrosesan HOG
gray_image = rgb2gray(image)

# Hitung fitur Histogram of Oriented Gradients (HOG)
# fd = fitur descriptor, hog_image = citra hasil visualisasi HOG
fd, hog_image = hog(gray_image,
                    orientations=8,           # jumlah orientasi gradien
                    pixels_per_cell=(16, 16), # ukuran cell (blok piksel) untuk menghitung histogram
                    cells_per_block=(1, 1),   # jumlah cell dalam satu blok normalisasi
                    visualize=True)           # minta visualisasi hasil HOG

# Membuat figure dengan 2 subplot untuk menampilkan gambar input dan hasil HOG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Tampilkan citra grayscale tanpa axis
ax1.axis('off')
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Input Image (Grayscale)')

# Rescale intensitas gambar HOG agar lebih jelas untuk visualisasi
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Tampilkan citra HOG yang sudah di-rescale tanpa axis
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

# Atur layout agar rapi dan tampilkan plot
plt.tight_layout()
plt.show()
