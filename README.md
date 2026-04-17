# 🌾 Agriculture Production Optimization & Yield Prediction

A Machine Learning-powered web application that predicts crop yields based on environmental factors like rainfall, temperature, and pesticide usage.

## 🚀 Key Features
- **High Accuracy:** Trained a Decision Tree Regressor achieving an **R² score of 0.98**.
- **Interactive UI:** Built with Flask, allowing users to input variables and get real-time predictions.
- **Data-Driven:** Analyzes historical data across 10+ countries and various crop types.

## 🛠️ Technical Stack
- **Languages:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **Web Framework:** Flask
- **Deployment:** Render / GitHub
- **Model Serialization:** Pickle

## 📊 The Model
The project utilizes a `Decision Tree Regressor` to handle the non-linear relationships between environmental factors and crop output. The data was preprocessed using a `ColumnTransformer` with `StandardScaler` for numerical data and `OneHotEncoder` for categorical data (Area and Item).

## 📂 Project Structure
- `app.py`: Flask application logic.
- `yield_df.csv`: The processed dataset.
- `dtr.pkl`: The trained Decision Tree model.
- `preprocessor.pkl`: The fitted scikit-learn preprocessing pipeline.
- `Agriculture Production Optimization.ipynb`: The original data analysis and training notebook.

## ⚙️ How to Run Locally
1. Clone the repo: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
4. Access at `http://127.0.0.1:8001`
