import pickle
import numpy as np

model = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/lung_model.pkl", "rb"))
features = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/features.pkl", "rb"))

# High risk dummy
high_risk = np.array([[8]*25])
for i, f in enumerate(features):
    if f == "Age": high_risk[0][i] = 60
    if f == "Gender": high_risk[0][i] = 1 # Male
    if f in ["index", "Patient Id"]: high_risk[0][i] = 0

p_high = model.predict_proba(high_risk)[0]
print("High Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_high):
    print(i, prob)

# Low risk dummy
low_risk = np.array([[1]*25])
for i, f in enumerate(features):
    if f == "Age": low_risk[0][i] = 20
    if f == "Gender": low_risk[0][i] = 2 # Female
    if f in ["index", "Patient Id"]: low_risk[0][i] = 0

p_low = model.predict_proba(low_risk)[0]
print("\nLow Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_low):
    print(i, prob)
