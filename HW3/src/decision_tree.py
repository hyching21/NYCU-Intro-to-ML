"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or self.gini(y) == 0:
            return Node(value=np.bincount(y).argmax())
        feature_index, threshold = find_best_split(X, y)
        if feature_index is None:
            return Node(value=np.bincount(y).argmax())
        left = X[:, feature_index] <= threshold
        right = ~left
        if len(y[left]) == 0 or len(y[right]) == 0:
            return Node(value=np.bincount(y).argmax())
        left_subtree = self._grow_tree(X[left], y[left], depth + 1)
        right_subtree = self._grow_tree(X[right], y[right], depth + 1)
        return Node(feature=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        node = tree_node
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def gini(self, data):
        return gini(data)

    def entro(self, data):
        return entropy(data)

    # def plot_feature_importance(self, columns, fpath):
    #     feature_importance = np.zeros(len(columns))

    #     def dfs(node):
    #         if node.value is not None:
    #             return
    #         feature_importance[node.feature] += 1
    #         dfs(node.left)
    #         dfs(node.right)
    #     dfs(self.tree)

    #     indices = np.argsort(feature_importance)[::-1]
    #     sorted_importance = feature_importance[indices]
    #     sorted_names = [columns[i] for i in indices]

    #     plt.figure(figsize=(10, 6))
    #     plt.bar(range(len(sorted_importance)), sorted_importance, align="center", alpha=0.7, color='skyblue')
    #     plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right", fontsize=10)
    #     plt.xlabel("Features", fontsize=12)
    #     plt.ylabel("Importance", fontsize=12)
    #     plt.title("Feature Importance")
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig(fpath)
    #     plt.close()
    def plot_feature_importance(self, columns, X, y):
        if self.tree is None:
            raise ValueError("The tree has not been fitted.")
        feature_importances = {col: 0 for col in columns}

        def traverse(node, X, y):
            if node.value is not None:
                return
            parent_gini = self.gini(y)
            left_indices = X[:, node.feature] <= node.threshold
            right_indices = ~left_indices
            y_left, y_right = y[left_indices], y[right_indices]
            if len(y_left) == 0 or len(y_right) == 0:
                return
            left_gini = self.gini(y_left)
            right_gini = self.gini(y_right)
            weighted_gini = (len(y_left) * left_gini + len(y_right) * right_gini) / len(y)
            gini_reduction = parent_gini - weighted_gini
            feature_importances[columns[node.feature]] += gini_reduction
            traverse(node.left, X[left_indices], y_left)
            traverse(node.right, X[right_indices], y_right)
        traverse(self.tree, X, y)
        # Sort
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_values = zip(*sorted_importances)
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_names, sorted_values, align="center", alpha=0.7, color='skyblue')
        plt.title("Feature Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_indices = X[:, feature_index] <= threshold
    right_indices = ~left_indices
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]


# Find the best split for the dataset
def find_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            # Calculate Gini index for the split
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold


def gini(y):
    gini_index = 1 - np.sum((np.bincount(y) / len(y)) ** 2)
    return gini_index


def entropy(y):
    prob = np.bincount(y) / len(y) + 1e-10
    prob[prob == 0] = 1
    entropy = -np.sum(prob * np.log2(prob))
    return entropy
