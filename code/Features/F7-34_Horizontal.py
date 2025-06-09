import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

row_counts_by_digit = {i: [[] for _ in range(28)] for i in range(10)}

for image, label in zip(src_images, src_labels):
    for row in range(28):
        count = np.sum(image[row, :] > 0)
        row_counts_by_digit[label][row].append(count)

fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axs = axs.flatten()

for digit in range(10):
    avg_counts = [np.mean(row_counts_by_digit[digit][row]) for row in range(28)]
    axs[digit].plot(range(28), avg_counts, marker='o', color='mediumseagreen')
    axs[digit].set_title(f'Digit {digit}')
    axs[digit].set_xlabel('Row (0–27)')
    axs[digit].set_xticks(range(0, 28, 4))
    axs[digit].grid(axis='y', linestyle='--', alpha=0.6)

axs[0].set_ylabel('Average Non-Zero Pixel Count')
axs[5].set_ylabel('Average Non-Zero Pixel Count') 
plt.suptitle('Feature 7-34: Average Non-Zero Pixels Per Row for Each Digit')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
