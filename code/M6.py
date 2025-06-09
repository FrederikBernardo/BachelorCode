""" Decision tree """
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from FeatureExtractionM2 import FeatureExtraction
from MNISTPy import read_mnist_images, read_mnist_labels
from FindBest import find_best_split


train_src_images = read_mnist_images('MNIST/train-images.idx3-ubyte')
train_src_labels = read_mnist_labels('MNIST/train-labels.idx1-ubyte')

test_src_images = read_mnist_images('MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_src_labels = read_mnist_labels('MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

train_features = np.array([FeatureExtraction(image) for image in train_src_images])
train_labels = np.array(train_src_labels)

test_features = np.array([FeatureExtraction(image) for image in test_src_images])
test_labels = np.array(test_src_labels)

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

def build_tree(features, y, depth=0, max_depth=10, num_thresholds=25, min_samples_split=500):
    if depth == max_depth or len(y) < min_samples_split or len(set(y)) == 1:
        return Node(prediction=Counter(y).most_common(1)[0][0])

    best_feat_idx, best_threshold, best_mi = find_best_split(features, y, num_thresholds)

    if best_feat_idx is None or best_mi < 1e-6:
        return Node(prediction=Counter(y).most_common(1)[0][0])

    left_mask = features[:, best_feat_idx] <= best_threshold
    right_mask = ~left_mask
    
    left_subtree = build_tree(
        features[left_mask], y[left_mask], 
        depth+1, max_depth, min_samples_split, num_thresholds
    )
    
    right_subtree = build_tree(
        features[right_mask], y[right_mask],
        depth+1, max_depth, min_samples_split, num_thresholds
    )

    return Node(
        feature_idx=best_feat_idx,
        threshold=best_threshold,
        left=left_subtree,
        right=right_subtree
    )

def predict_tree(x, node):
    if node.prediction is not None:
        return node.prediction
    if x[node.feature_idx] <= node.threshold:
        return predict_tree(x, node.left)
    else:
        return predict_tree(x, node.right)

def predict(X, tree_root):
    return np.array([predict_tree(x, tree_root) for x in X])

tree = build_tree(train_features, train_labels)

def accuracy(true_labels, pred_labels):
    return np.mean(true_labels == pred_labels) * 100

train_preds = predict(train_features, tree)
test_preds = predict(test_features, tree)

sns.heatmap(confusion_matrix(train_labels, train_preds), annot=True, fmt='d', cmap='Blues')
plt.title("M6: Confusion Matrix - Training Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Accuracy:", accuracy_score(train_labels, train_preds))
print("\nClassification Report:\n", classification_report(train_labels, train_preds))
print("\nConfusion Matrix:\n", confusion_matrix(train_labels, train_preds))

sns.heatmap(confusion_matrix(test_labels, test_preds), annot=True, fmt='d', cmap='Blues')
plt.title("M6: Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Accuracy:", accuracy_score(test_labels, test_preds))
print("\nClassification Report:\n", classification_report(test_labels, test_preds))
print("\nConfusion Matrix:\n", confusion_matrix(test_labels, test_preds))