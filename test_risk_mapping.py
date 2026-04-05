import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("model/lung_model.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

# High risk dummy
high_risk = {f: 8 for f in features}
for f in features:
    if f == "Age": high_risk[f] = 60
    if f == "Gender": high_risk[f] = 1 # Male
    if f in ["index", "Patient Id"]: high_risk[f] = 0

df_high = pd.DataFrame([high_risk[f] for f in features]).T
df_high.columns = features

p_high = model.predict_proba(df_high)[0]
print("High Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_high):
    print(i, prob)

# Low risk dummy
low_risk = {f: 1 for f in features}
for f in features:
    if f == "Age": low_risk[f] = 20
    if f == "Gender": low_risk[f] = 2 # Female
    if f in ["index", "Patient Id"]: low_risk[f] = 0

df_low = pd.DataFrame([low_risk[f] for f in features]).T
df_low.columns = features

p_low = model.predict_proba(df_low)[0]
print("\nLow Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_low):
    print(i, prob)
