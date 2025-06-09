import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

volume_by_label = {i: [] for i in range(10)}

for image, label in zip(src_images, src_labels):
    volume = np.sum(image > 0)
    volume_by_label[label].append(volume)


plt.figure(figsize=(10, 6))
plt.boxplot([volume_by_label[digit] for digit in range(10)], labels=range(10))
plt.xlabel("Digit")
plt.ylabel("Nonzero Pixels")
plt.title("Feature 2: Distribution of volume for MNIST Digits")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()


