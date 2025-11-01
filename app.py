import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ------------------- LOAD DATA -------------------
df = pd.read_csv('yield_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Remove string errors in rainfall column
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

df = df.drop(df[df['average_rain_fall_mm_per_year'].apply(isStr)].index)

df.drop_duplicates(inplace=True)

# ------------------- FEATURE SELECTION -------------------
df = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ------------------- DATA SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

# ------------------- PREPROCESSING -------------------
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OHE', ohe, [4, 5]),
    ],
    remainder='passthrough'
)

X_train_processed = preprocesser.fit_transform(X_train)
X_test_processed = preprocesser.transform(X_test)

# ------------------- MODEL TRAINING -------------------
dtr = DecisionTreeRegressor()
dtr.fit(X_train_processed, y_train)

y_pred = dtr.predict(X_test_processed)

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------- SAVE MODEL -------------------
pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocessor.pkl', 'wb'))

print("\n✅ Model and Preprocessor Saved Successfully as dtr.pkl & preprocessor.pkl")

# ------------------- PREDICTION FUNCTION -------------------
def predict_yield(Year, Rain, Pesticides, Temp, Area, Crop):
    data = np.array([[Year, Rain, Pesticides, Temp, Area, Crop]], dtype=object)
    data_transformed = preprocesser.transform(data)
    prediction = dtr.predict(data_transformed)
    return prediction[0]

# Example test call:
tes
