🍽️ Restaurant Rating Prediction

A machine learning project that predicts the aggregate rating of a restaurant using features such as location, cuisine, pricing, customer votes, and service availability.

🎯 Objective

Build a regression model that learns from restaurant data and predicts the expected Aggregate Rating for a restaurant.

🔄 Workflow

Dataset
   ↓
Data Cleaning
   ↓
Preprocessing & Feature Engineering
   ↓
Train / Test Split
   ↓
Regression Models
   ↓
Model Evaluation
   ↓
Prediction & Feature Analysis

📊 Dataset

The dataset contains restaurant information such as:

Restaurant Name

City & Locality

Cuisines

Average Cost for Two

Price Range

Table Booking

Online Delivery

Votes

Aggregate Rating

Target Variable: Aggregate rating

🤖 Machine Learning

The project uses regression techniques to predict restaurant ratings.

Models can include:

Linear Regression

Decision Tree Regression

Random Forest Regression

Evaluation Metrics

MAE — Mean Absolute Error

MSE — Mean Squared Error

RMSE — Root Mean Squared Error

R² Score

📁 Project Structure

Restaurant_Rating_Prediction/
│
├── data/
├── notebooks/
├── outputs/
├── src/
├── venv/
├── .gitignore
├── main.py
├── model_audit.py
├── restaurants.csv
├── requirements.txt
└── README.md

venv/ is a local virtual environment and should not be uploaded to GitHub.

🛠️ Technologies

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Jupyter Notebook

🚀 Installation & Usage

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/Restaurant_Rating_Prediction.git
cd Restaurant_Rating_Prediction

2. Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the project

python main.py

Generated predictions, evaluation results, and visualizations are stored in the outputs/ directory.

📌 Key Features

Data preprocessing

Missing-value handling

Categorical encoding

Feature engineering

Regression model training

Model comparison

Rating prediction

Feature importance analysis

Result visualization

🔮 Future Improvements

Hyperparameter tuning

Cross-validation

Advanced ensemble models

NLP-based review analysis

Explainable AI using SHAP

Web application deployment

👨‍💻 Author

Jibin
B.Tech Computer Science Engineering Student

⭐ If you find this project useful, consider giving it a star!
