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

    return [boundary, volume, descaling_squared, descaling_canvas, max_length, approximated_points]