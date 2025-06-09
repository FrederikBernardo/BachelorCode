import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MNISTPy import read_mnist_images, read_mnist_labels

src_images = read_mnist_images('../MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('../MNIST/train-labels.idx1-ubyte')

target_digit = 5
for image, label in zip(src_images, src_labels):
    if label == target_digit:
        sample_image = image
        break

fig, axs = plt.subplots(4, 1, figsize=(4, 28))

for i in range(4):
    row_start = i * 7
    row_end = row_start + 7
    axs[i].imshow(sample_image[row_start:row_end, :], cmap='gray', aspect='auto')
    axs[i].axis('off')

plt.suptitle(f'Digit {target_digit} – Split into 4 Slices (7 Rows Each)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
