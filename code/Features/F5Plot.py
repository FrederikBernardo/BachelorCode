import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

def image_processing_laplacian(images, labels, index):
    fig, axes = plt.subplots(len(index), 3, figsize=(10, 8))
    fig.tight_layout()

    kernel_size = 3
    ddepth_8U = cv.CV_8U

    for row, idx in enumerate(index):
        src_image = images[idx]
        gaussian_blur_images = cv.GaussianBlur(src_image, (3,3), 0)
        laplacian_image_8U = cv.Laplacian(gaussian_blur_images, ddepth_8U, ksize=kernel_size)

        contours, _ = cv.findContours(src_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        max_length = 0
        longest_line = None

        for contour in contours:
            approx_contour = cv.approxPolyDP(contour, epsilon=1, closed=False)

            for i in range(len(approx_contour) - 1):
                point1 = approx_contour[i][0]
                point2 = approx_contour[i + 1][0]

                length = np.linalg.norm(point1 - point2)

                if max_length < length:
                    max_length = length
                    longest_line = (point1, point2)

        laplacian_rgb_copy = cv.cvtColor(laplacian_image_8U, cv.COLOR_GRAY2BGR)
        if longest_line != None:
            point1, point2 = longest_line
            cv.line(laplacian_rgb_copy, tuple(point1), tuple(point2), (0, 255, 0), 1)
        
        canvas = np.zeros_like(src_image)

        axes[row, 0].imshow(src_image, cmap='gray')
        axes[row, 0].set_title(f"Digit {labels[idx]} - Source image")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(cv.cvtColor(laplacian_rgb_copy, cv.COLOR_BGR2RGB))
        axes[row, 1].set_title(f"Digit {labels[idx]} - Longest Line: {max_length:.2f}px")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(cv.drawContours(canvas, approx_contour, -1, (255), 1), cmap='gray')
        axes[row, 2].set_title(f"Digit {labels[idx]} - Contour approximation")
        axes[row, 2].axis('off')

    plt.show()

index = [3, 223, 17]
plotting = image_processing_laplacian(src_images, src_labels, index)