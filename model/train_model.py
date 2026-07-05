import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
df = pd.read_csv('clean_hdi.csv')

X = df[['life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita']]
y = df['hdi_tier']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

os.makedirs('model', exist_ok=True)

with open('model/hdi_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Success! Model files created in the 'model' folder.")