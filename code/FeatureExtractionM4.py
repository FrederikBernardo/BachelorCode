""" This FeatureExtraction is simplified in relation to what
features were the least informative in regards to the Coefficient calculations

To compute the coefficients we used the LinearSVC() function which uses a different approach.
This is OvR strategy and therefore it only paints a picture of valuable coefficients
(weights of each feature) based on a different calculation than SVC(kernel='linear') found in M1 and M2

Scope: Test the accuracy of the simplified feature extraction omitting less informative features """

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

def FeatureExtraction(image):
    """ Feature 1 - Edge Area"""
    edges = cv.Canny(image, 50, 150)
    boundary = np.sum(edges == 255)

    """ Feature 2 - Volume"""
    volume = np.sum(image > 0)

    """ Feature 3 - Descaling Squared"""
    descaling_squared = boundary / np.sqrt(volume)

    """ Feature 4 - Descaling Removed inside circle"""
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    canvas = (cv.drawContours(np.zeros_like(image), contours, -1, (255), 1))
    boundary_canvas = np.sum(canvas == 255)
    
    descaling_canvas = boundary_canvas / volume

    """ Feature 5 & Feature 6 - Longest line and approximated point"""
    max_length = 0
    for contour in contours:
        approx_contour = cv.approxPolyDP(contour, epsilon=1, closed=False)
        approximated_points = len(approx_contour)

        for i in range(len(approx_contour) - 1):
                point1 = approx_contour[i][0]
                point2 = approx_contour[i+1][0]

                length = np.linalg.norm(point1 - point2)
                if max_length < length:
                    max_length = length

    """ Feature 7-34 - Horizontal Segmentation features: Expanded """
    hori_slice_height = image.shape[0] // 28
    hori_slice_counts = []
    for i in range(28):
        if i in [0, 1, 16]:
            continue
        hori_slice_image = image[i * hori_slice_height:(i + 1) * hori_slice_height, :]
        hori_nonzero_count = np.sum(hori_slice_image > 0)
        hori_slice_counts.append(hori_nonzero_count)

    """ Feature 35-62 - Vertical Segmentation Features: Expanded"""
    vert_slice_height = image.shape[1] // 28
    vert_slice_counts = []
    for i in range(28):
        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27]:
            continue
        vert_slice_image = image[:, i * vert_slice_height:(i + 1) * vert_slice_height]
        vert_nonzero_count = np.sum(vert_slice_image > 0)
        vert_slice_counts.append(vert_nonzero_count)

    return [boundary, descaling_squared, descaling_canvas, max_length, approximated_points, *hori_slice_counts, *vert_slice_counts]