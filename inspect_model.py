import pickle
import numpy as np

model = pickle.load(open("model/lung_model.pkl", "rb"))

print("Number of features:", model.n_features_in_)

# Check some thresholds in the first tree to see the range of values
tree = model.estimators_[0].tree_
thresholds = tree.threshold[tree.threshold != -2] # -2 is for leaves
print("Sample Thresholds (Min/Max):", np.min(thresholds), np.max(thresholds))

# If thresholds are around 1, 2, 3, then it's a 1-8 or 1-3 scale.
# If they are around 0.5, then it's a 0/1 binary scale.
