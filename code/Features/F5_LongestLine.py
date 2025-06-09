import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')
    

""" Image processing function: Source Image """

def image_processing(images, labels):

    longest_line_by_label = {i: [] for i in range(10)}

    for image, label in zip(images, labels) :
        src_image = image

        contours, _ = cv.findContours(src_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        max_length = 0

        for contour in contours:
            approx_contour = cv.approxPolyDP(contour, epsilon=1, closed=False)

            for i in range(len(approx_contour) - 1):
                point1 = approx_contour[i][0]
                point2 = approx_contour[i+1][0]

                length = np.linalg.norm(point1 - point2)

                if max_length < length:
                    max_length = length
        
        longest_line_by_label[label].append(max_length)

    return longest_line_by_label

results = image_processing(src_images, src_labels)

plt.figure(figsize=(10, 6))
plt.boxplot([results[digit] for digit in range(10)], labels=range(10))
plt.xlabel("Digit")
plt.ylabel("Length of Longest Line")
plt.title("Feature 5: Longest line pr digit")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
