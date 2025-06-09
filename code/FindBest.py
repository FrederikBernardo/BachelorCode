import numpy as np
from collections import Counter

def entropy_from_counts(counts):
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=float) / total
    return -np.sum(probs * np.log2(probs + 1e-9))

def entropy(labels):
    return entropy_from_counts(Counter(labels))

def joint_entropy(labels, binary_split):
    joint = list(zip(labels, binary_split))
    return entropy_from_counts(Counter(joint))

def mutual_information(labels, binary_split):
    return entropy(labels) + entropy(binary_split) - joint_entropy(labels, binary_split)

def find_best_split(features, labels, num_thresholds=25):
    best_mi = -np.inf
    best_feat_idx = None
    best_threshold = None

    num_features = features.shape[1]

    for feat_idx in range(num_features):
        feat_values = features[:, feat_idx]
        min_val, max_val = np.min(feat_values), np.max(feat_values)
        thresholds = np.linspace(min_val, max_val, num=num_thresholds)

        for threshold in thresholds:
            binary_split = feat_values > threshold
            mi = mutual_information(labels, binary_split)

            if mi > best_mi:
                best_mi = mi
                best_feat_idx = feat_idx
                best_threshold = threshold

    return best_feat_idx, best_threshold, best_mi
