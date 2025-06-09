import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

""" Used for finding numbers """
def numberfinder (labels,  num):
    return [i for i, label in enumerate(labels) if label == num]

results = numberfinder(src_labels, 1)

index = [1, 18, 31, 4]

def contourdraw (images, labels, index):
    fig, axes = plt.subplots(len(index), 3, figsize=(8,8))
    fig.tight_layout()

    for row, idx in enumerate(index):
        src_image = images[idx]
        canny_image_edges = cv.Canny(src_image, 50, 150)

        contours, _ = cv.findContours(src_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        canvas = (cv.drawContours(np.zeros_like(src_image), contours, -1, (255), 1))

        boundary_contours = np.sum(canvas == 255)
        boundary_canny = np.sum(canny_image_edges == 255)
        difference = boundary_canny - boundary_contours
        print("\n" + (50 * "="))
        print(f"Boundary of each digit using canny: {labels[idx]} = {boundary_canny} ")
        print(f"Boundary of each digit using contours: {labels[idx]} = {boundary_contours}")
        print(f"Difference of boundary: {difference}")
        print((50 * "="))

        axes[row, 0].imshow(src_image, cmap='gray')
        axes[row, 0].set_title(f"Digit = {labels[idx]} - Source")

        axes[row, 1].imshow(canny_image_edges, cmap='gray')
        axes[row, 1].set_title(f"Digit = {labels[idx]} - Canny Edges")

        axes[row, 2].imshow(canvas, cmap='gray')
        axes[row, 2].set_title(f"Digit = {labels[idx]} - Contours")
    
    plt.show()

contourdraw(src_images,src_labels,index)