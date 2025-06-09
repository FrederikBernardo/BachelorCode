import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

boundary_by_label = {i: [] for i in range(10)}

""" We use the canny operator to find the boundary """

for image, label in zip(src_images, src_labels):
    edges = cv.Canny(image, 50, 150)
    boundary = np.sum(edges == 255)
    boundary_by_label[label].append(boundary)

plt.figure(figsize=(10, 6))
plt.boxplot([boundary_by_label[digit] for digit in range(10)], labels=range(10))
plt.xlabel('Digit')
plt.ylabel('Edge Area')
plt.title('Feature 1: Edge Area Distribution for Each Digit')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()