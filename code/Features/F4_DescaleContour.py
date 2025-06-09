import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

def contour_descale (images, labels):
    scaled_by_label = {i: [] for i in range(10)}

    for image, label in zip(images, labels):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        canvas = (cv.drawContours(np.zeros_like(image), contours, -1, (255), 1))
        boundary = np.sum(canvas == 255)
        volume = np.sum(image > 0)

        scale = boundary / volume
        scaled_by_label[label].append(scale)

    return scaled_by_label

results = contour_descale(src_images, src_labels)

plt.figure(figsize=(10, 6))
plt.boxplot([results[digit] for digit in range(10)], labels=range(10))
plt.xlabel('Digit')
plt.ylabel('Boundary Divided by volume')
plt.title('Feature 4 - Normalization')
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

