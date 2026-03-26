import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# Try XGBoost (if installed)
try:
    from xgboost import XGBClassifier
    use_xgb = True
except:
    use_xgb = False

# -------------------- LOAD DATA --------------------
df = pd.read_csv("dataset/lung_cancer.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------- DATA CLEANING --------------------

# Remove unwanted column if exists
if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

# Encode categorical columns (Yes/No → 1/0)
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -------------------- FEATURE & TARGET --------------------

target_column = "LUNG_CANCER"   # change if your dataset uses 'Level'

X = df.drop(target_column, axis=1)
y = df[target_column]

# Save feature names (VERY IMPORTANT for app)
feature_names = X.columns.tolist()

# -------------------- TRAIN TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- MODEL --------------------

if use_xgb:
    print("Using XGBoost Model")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss'
    )
else:
    print("Using Random Forest Model")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10
    )

# Train
model.fit(X_train, y_train)

# -------------------- EVALUATION --------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -------------------- SAVE MODEL --------------------

pickle.dump(model, open("model/lung_model.pkl", "wb"))

# Save feature names
pickle.dump(feature_names, open("model/features.pkl", "wb"))

print("Model saved successfully!")