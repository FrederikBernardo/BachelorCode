import matplotlib.pyplot as plt
import cv2 as cv
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

src_image = src_images[90]
src_label = src_labels[90]

kernel_size = 3

ddepth_8U = cv.CV_8U
ddepth_16S = cv.CV_16S

src_image_gauss = cv.GaussianBlur(src_image, (3, 3), 0)
laplacian_image_8U = cv.Laplacian(src_image, ddepth_8U, ksize=kernel_size)
laplacian_image_16S = cv.Laplacian(src_image, ddepth_16S, ksize=kernel_size)

sobel_x = cv.Sobel(src_image, cv.CV_64F, 1, 0, ksize=kernel_size)
sobel_y = cv.Sobel(src_image, cv.CV_64F, 0, 1, ksize=kernel_size)
sobel_combined = cv.magnitude(sobel_x, sobel_y)

canny_edges = cv.Canny(src_image_gauss, 50, 150)

plt.figure(figsize=(8,8))

plt.subplot(1, 4, 1)
plt.imshow(src_image, cmap='gray')
plt.title(f'Target label: {src_label}')

plt.subplot(1, 4, 2)
plt.imshow(laplacian_image_8U, cmap='gray')
plt.title('Laplacian: depth = 8U')

plt.subplot(1, 4, 3)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel')

plt.subplot(1, 4, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny')

plt.tight_layout()
plt.show()

