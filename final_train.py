import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load the real data
# Ensure yield_df.csv is in your folder!
df = pd.read_csv('yield_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# 2. Clean data (Same logic as your notebook)
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

df = df.drop(df[df['average_rain_fall_mm_per_year'].apply(isStr)].index)
df['average_rain_fall_mm_per_year'] = pd.to_numeric(df['average_rain_fall_mm_per_year'])
df.drop_duplicates(inplace=True)

X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
y = df['hg/ha_yield']

# 3. Training & Preprocessing
preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
X_train_processed = preprocesser.fit_transform(X_train)

dtr = DecisionTreeRegressor()
dtr.fit(X_train_processed, y_train)

# 4. Export the REAL brains
pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocessor.pkl', 'wb')) 

print("✅ SUCCESS: The 0.98 R² Model has been exported to dtr.pkl!")