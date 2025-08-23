import math, random
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------------
# 1. Utility Functions
# -------------------------------

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def info_gain(parent_labels, left_labels, right_labels):
    total = len(parent_labels)
    pL, pR = len(left_labels)/total, len(right_labels)/total
    return entropy(parent_labels) - (pL*entropy(left_labels) + pR*entropy(right_labels))

# -------------------------------
# 2. Decision Tree Class
# -------------------------------

class DecisionTree:
    def __init__(self, max_depth=None, min_size=1, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None
    
    def fit(self, X, y):
        data = [list(x) + [y[i]] for i, x in enumerate(X)]
        self.features_count = len(X[0])
        if self.n_features is None:
            self.n_features = self.features_count
        self.tree = self._build_tree(data, depth=0)
    
    def _best_split(self, dataset):
        parent_labels = [row[-1] for row in dataset]
        best_gain, best_feature, best_value, best_splits = 0, None, None, None
        features = random.sample(range(self.features_count), self.n_features)
        
        for feature in features:
            values = set(row[feature] for row in dataset)
            for val in values:
                left = [row for row in dataset if row[feature] <= val]
                right = [row for row in dataset if row[feature] > val]
                if not left or not right:
                    continue
                gain = info_gain(parent_labels, [r[-1] for r in left], [r[-1] for r in right])
                if gain > best_gain:
                    best_gain, best_feature, best_value, best_splits = gain, feature, val, (left, right)
        return best_feature, best_value, best_splits
    
    def _to_leaf(self, dataset):
        labels = [row[-1] for row in dataset]
        return Counter(labels).most_common(1)[0][0]
    
    def _build_tree(self, dataset, depth):
        labels = [row[-1] for row in dataset]
        
        if labels.count(labels[0]) == len(labels):  # pure node
            return labels[0]
        if self.max_depth and depth >= self.max_depth:
            return self._to_leaf(dataset)
        if len(dataset) <= self.min_size:
            return self._to_leaf(dataset)
        
        feature, value, splits = self._best_split(dataset)
        if not splits:
            return self._to_leaf(dataset)
        
        left, right = splits
        return {
            "feature": feature,
            "value": value,
            "left": self._build_tree(left, depth+1),
            "right": self._build_tree(right, depth+1)
        }
    
    def _predict_row(self, node, row):
        if not isinstance(node, dict):
            return node
        if row[node["feature"]] <= node["value"]:
            return self._predict_row(node["left"], row)
        else:
            return self._predict_row(node["right"], row)
    
    def predict(self, X):
        return [self._predict_row(self.tree, row) for row in X]

# -------------------------------
# 3. Random Forest Class
# -------------------------------

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_size=1, sample_size=None, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.trees = []
    
    def _subsample(self, X, y):
        n_samples = self.sample_size or len(X)
        indices = [random.randrange(len(X)) for _ in range(n_samples)]
        return [X[i] for i in indices], [y[i] for i in indices]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._subsample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions).T  # trees × samples → samples × trees
        final = []
        for row in predictions:
            final.append(Counter(row).most_common(1)[0][0])
        return final



# Load iris dataset
iris = load_iris()
X, y = iris.data.tolist(), iris.target.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForest(n_trees=10, max_depth=5, n_features=2)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Accuracy
accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
print("Random Forest Accuracy:", accuracy)
