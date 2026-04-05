import pickle
model = pickle.load(open("model/lung_model.pkl", "rb"))
print("Classes:", model.classes_)
