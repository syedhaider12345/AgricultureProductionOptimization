from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# 1. Load the "Brains"
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# 2. Hard-coded lists for the dropdowns
countries = ['India', 'Pakistan', 'Bangladesh', 'Australia', 'Brazil', 'Egypt', 'Japan', 'Mexico', 'South Africa', 'United Kingdom', 'United States']
crops = ['Maize', 'Potatoes', 'Rice', 'Sorghum', 'Soybeans', 'Wheat', 'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']

@app.route('/')
def index():
    return render_template('index.html', countries=countries, crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', countries=countries, crops=crops, prediction=prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True, port=8001)