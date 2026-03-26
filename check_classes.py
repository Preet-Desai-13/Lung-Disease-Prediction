import pickle
model = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/lung_model.pkl", "rb"))
print("Classes:", model.classes_)
