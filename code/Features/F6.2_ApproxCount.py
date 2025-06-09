import matplotlib.pyplot as plt
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')


def approximationCount(images, labels):
    approx_by_label = {i: [] for i in range(10)}
    
    for image, label in zip(images, labels):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for contour in contours:
            approx = cv.approxPolyDP(contour, epsilon=2, closed=False)
            count_approx = len(approx)

            approx_by_label[label].append(count_approx)

    return approx_by_label

results = approximationCount(src_images, src_labels)

plt.figure(figsize=(10, 6))
plt.boxplot([results[digit] for digit in range(10)], labels=range(10))
plt.xlabel('Digit')
plt.ylabel('Total approximated points')
plt.title('Feature 6 - Approximated points (Epsilon = 2)')
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
