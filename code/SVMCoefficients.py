from MNISTPy import read_mnist_images, read_mnist_labels
from FeatureExtractionM3 import FeatureExtraction

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

train_src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
train_src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

features = [FeatureExtraction(image) for image in train_src_images]
X_train = np.array(features)
y_train = np.array(train_src_labels)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LinearSVC()
model.fit(X_train, y_train)

coef = model.coef_
print("Coefficient matrix shape:", coef.shape)


num_classes = coef.shape[0]
num_features = coef.shape[1]
feature_names = [f"F{i+1}" for i in range(num_features)]

avg_abs_coef = np.mean(np.abs(coef), axis=0)

feature_importance = list(zip(feature_names, avg_abs_coef))

feature_importance.sort(key=lambda x: x[1], reverse=True)

print("Top features by average absolute coefficient:")
for rank, (fname, importance) in enumerate(feature_importance, 1):
    print(f"{rank:2d}. {fname}: {importance:.4f}")

for i in range(num_classes):
    print(f"Coefficients for digit {i}:", coef[i])

plt.figure(figsize=(12, 8))
for i in range(num_classes):
    plt.scatter(range(num_features), coef[i], s=20, label=f"Digit {i}", alpha=0.7)
plt.xticks(ticks=range(num_features), labels=feature_names, rotation=45)
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("SVM Coefficients for Each Digit Class (Dot Plot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



