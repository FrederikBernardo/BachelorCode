import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

scaled_by_label = {i: [] for i in range(10)}

""" Divide squared """

for image, label in zip(src_images, src_labels):
    edges = cv.Canny(image, 50, 150)
    boundary = np.sum(edges == 255)
    volume = np.sqrt(np.sum(image > 0))
    scaling = boundary / volume
    scaled_by_label[label].append(scaling)

plt.figure(figsize=(10, 6))
plt.boxplot([scaled_by_label[digit] for digit in range(10)], labels=range(10))
plt.xlabel('Digit')
plt.ylabel('Boundary Divided by volume')
plt.title('Feature 3: Normalization (Squared)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()