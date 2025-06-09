import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

slice_counts_by_label = [{i: [] for i in range(10)} for _ in range(4)]

for image, label in zip(src_images, src_labels):
    slice_height = image.shape[0] // 4
    for i in range(4):
        slice_image = image[i*slice_height:(i+1)*slice_height, :]
        nonzero_count = np.sum(slice_image > 0)
        slice_counts_by_label[i][label].append(nonzero_count)

fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axs = axs.flatten()

for digit in range(10):
    avg_counts = [np.mean(slice_counts_by_label[slice_idx][digit]) for slice_idx in range(4)]
    axs[digit].plot(range(1, 5), avg_counts, marker='o', color='mediumseagreen')
    axs[digit].set_title(f'Digit {digit}')
    axs[digit].set_xlabel('Slice (1–4)')
    axs[digit].set_xticks([1, 2, 3, 4])
    axs[digit].grid(axis='y', linestyle='--', alpha=0.6)

axs[0].set_ylabel('Average Non-Zero Pixel Count')
axs[5].set_ylabel('Average Non-Zero Pixel Count')
plt.suptitle('Average Non-Zero Pixels per Horizontal Slice for Each Digit')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()