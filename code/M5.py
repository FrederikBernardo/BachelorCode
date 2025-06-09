""" SVM OvO with simplified features (MI)"""

from MNISTPy import read_mnist_images, read_mnist_labels
from FeatureExtractionM5 import FeatureExtraction

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


train_src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
train_src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

test_src_images = read_mnist_images('MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_src_labels = read_mnist_labels('MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

features = [FeatureExtraction(image) for image in train_src_images]

X_train = np.array(features)
y_train = np.array(train_src_labels)

X_test = np.array([FeatureExtraction(image) for image in test_src_images])
y_test = np.array(test_src_labels)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_train_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('M5: Confusion Matrix - 20 Features removed')
plt.show()