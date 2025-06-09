import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from MNISTPy import read_mnist_images, read_mnist_labels
from FeatureExtractionM3 import FeatureExtraction

src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

X = [FeatureExtraction(img) for img in src_images]
y = src_labels

X_np = np.array(X)
y_np = np.array(y)

mi_scores = mutual_info_regression(X_np, y_np, discrete_features=False)

feature_names = [f"F{i+1}" for i in range(len(mi_scores))]

ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

print("Ranked Mutual Information Scores:")
for rank, (name, score) in enumerate(ranked, start=1):
    print(f"{rank:2d}. {name:5s}: MI = {score:.4f}")

plt.figure(figsize=(15, 6))
plt.bar(feature_names, mi_scores, color='teal')
plt.title("Mutual Information between Features and Digit Label")
plt.ylabel("Mutual Information")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
