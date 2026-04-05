import pickle
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

model = pickle.load(open('model/lung_model.pkl', 'rb'))
features = pickle.load(open('model/features.pkl', 'rb'))

# Test data
test_data = {f: 2 for f in features}
test_data['Age'] = 30
test_data['Gender'] = 1
test_data['Smoking'] = 7

df = pd.DataFrame([test_data[f] for f in features]).T
df.columns = features

probs = model.predict_proba(df)[0]
p_idx = int(np.argmax(probs))
perc = probs[p_idx] * 100

print('Prediction:', p_idx, perc)

# Try to create PDF
try:
    c = canvas.Canvas('test.pdf', pagesize=letter)
    c.drawString(100, 750, 'Test PDF')
    c.save()
    print('PDF created successfully')
except Exception as e:
    print('PDF error:', e)