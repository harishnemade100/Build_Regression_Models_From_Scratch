import numpy as np
import pandas as pd
from math import log2

# Example dataset
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Windy": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
print(df.head())



def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    entropy_val = 0
    for i in range(len(values)):
        p = counts[i] / np.sum(counts)
        entropy_val += -p * log2(p)
    return entropy_val



def info_gain(data, split_attribute, target_name="PlayTennis"):
    # Total entropy
    total_entropy = entropy(data[target_name])
    
    # Values of the splitting attribute
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    
    # Weighted entropy after split
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute] == vals[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target_name])
    
    # Information Gain
    gain = total_entropy - weighted_entropy
    return gain


def decision_tree(data, original_data, features, target_name="PlayTennis", parent_class=None):
    # If all target values have same class -> return that class
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]

    # If dataset is empty -> return mode of original data
    elif len(data) == 0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name], return_counts=True)[1])]
    
    # If features are empty -> return mode of parent
    elif len(features) == 0:
        return parent_class
    
    # Else: build tree
    else:
        # Majority class at current node
        parent_class = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        
        # Choose best feature
        gains = [info_gain(data, f, target_name) for f in features]
        best_feature = features[np.argmax(gains)]
        
        # Create tree root
        tree = {best_feature: {}}
        
        # Remove used feature
        remaining_features = [f for f in features if f != best_feature]
        
        # For each value of best feature → build subtree
        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            subtree = decision_tree(subset, df, remaining_features, target_name, parent_class)
            tree[best_feature][value] = subtree
        
        return tree

features = df.columns[:-1]  # All except target
tree = decision_tree(df, df, features.tolist())
print("Decision Tree:\n", tree)


def predict(query, tree, default="Yes"):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
 

 # Test example
query = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Windy": False}
print("Prediction:", predict(query, tree))


