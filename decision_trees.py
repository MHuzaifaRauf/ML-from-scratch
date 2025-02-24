"""
The implementation is based on this article: https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
"""

import numpy as np
from math import log2
from collections import Counter


def entropy(y):
    """Calculate the entropy of label array y."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * log2(p) for p in ps if p > 0])


class Node:
    """A node in the decision tree."""

    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, *, value=None
    ):
        # For decision node: feature_index & threshold are set; left and right are child nodes.
        # For leaf node: value is set.
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        """
        Initialize the decision tree.

        Parameters:
        - max_depth: maximum depth of the tree.
        - min_samples_split: minimum number of samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Fit the decision tree to data X and labels y."""
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        # Stopping criteria
        if (
            num_samples < self.min_samples_split
            or len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold, best_gain = None, None, -1
        best_left_indices, best_right_indices = None, None
        parent_entropy = entropy(y)

        for feature_index in range(num_features):
            # Consider unique values as candidate thresholds
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split data based on threshold
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # avoid useless splits

                # Compute weighted entropy for children
                left_entropy = entropy(y[left_indices])
                right_entropy = entropy(y[right_indices])
                n = len(y)
                n_left, n_right = len(left_indices), len(right_indices)
                child_entropy = (n_left / n) * left_entropy + (
                    n_right / n
                ) * right_entropy
                info_gain = parent_entropy - child_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        # If no significant gain, return a leaf node.
        if best_gain < 1e-6:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Recursively build the left and right subtrees.
        left_subtree = self._build_tree(
            X[best_left_indices, :], y[best_left_indices], depth + 1
        )
        right_subtree = self._build_tree(
            X[best_right_indices, :], y[best_right_indices], depth + 1
        )

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _most_common_label(self, y):
        """Return the most common label in y."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """Predict class labels for samples in X."""
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the tree recursively to make a prediction for a single sample."""
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
