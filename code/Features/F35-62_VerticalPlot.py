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

fig, axs = plt.subplots(1, 28, figsize=(28, 4), facecolor='white')

for i in range(28):
    axs[i].imshow(sample_image[:, i:i+1], cmap='gray', aspect='auto')
    axs[i].axis('off')
    axs[i].set_facecolor('white')

plt.suptitle(f'Digit {target_digit} – Split into 28 Column Slices', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

