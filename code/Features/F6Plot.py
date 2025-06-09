import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

idx = 9
src_image = src_images[idx]
epsilons = [1, 2, 3, 4]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
fig.suptitle(f"Digit {src_labels[idx]} - Approximation Points for Different ε", fontsize=14)

for i, epsilon in enumerate(epsilons):
    canvas = np.zeros_like(src_image)

    contours, _ = cv.findContours(src_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        approx_contour = cv.approxPolyDP(contour, epsilon=epsilon, closed=False)

    axes[i].imshow(cv.drawContours(canvas, approx_contour, -1, (255), 1), cmap='gray')
    axes[i].set_title(f"ε = {epsilon}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()