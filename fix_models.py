import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Create placeholder model
dtr = DecisionTreeRegressor()
dtr.fit(np.random.rand(10, 6), np.random.rand(10))

# Create placeholder preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(handle_unknown='ignore'), [4, 5]),
    ],
    remainder='passthrough'
)
X_dummy = np.array([[2013, 1000, 2000, 20, 'India', 'Maize']], dtype=object)
preprocessor.fit(X_dummy)

# Save them
pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))
print("✅ Files recreated!")