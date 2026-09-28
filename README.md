🍽️ Restaurant Rating Prediction

An end-to-end machine learning project for predicting restaurant
aggregate ratings using restaurant characteristics, pricing, customer
engagement, location, cuisines, and service-related features.

📌 Table of Contents

Project Overview

Problem Statement

Objectives

Dataset

Features

Target Variable

Machine Learning Workflow

Data Preprocessing

Models

Model Evaluation

Feature Analysis

Project Structure

Technologies Used

Installation

How to Run

Outputs

Model Audit

Results

Challenges and Limitations

Future Improvements

Learning Outcomes

Internship Task Mapping

GitHub Information

Author

Disclaimer

🔎 Project Overview

Restaurant ratings can be influenced by many factors, including
location, cuisine, pricing, customer engagement, table booking, online
delivery, and other restaurant characteristics.

The objective of this project is to build a supervised machine
learning regression system that predicts a restaurant's Aggregate
Rating from the available restaurant attributes.

The project follows an end-to-end workflow:

Restaurant Dataset
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Feature Analysis
       ↓
Predictions & Results

🎯 Problem Statement

Develop a machine learning model that can predict the aggregate rating
of a restaurant using other available restaurant attributes.

Since the target is a continuous numerical value, the task is formulated
as a regression problem.

The system is designed to:

Load the restaurant dataset.

Inspect the dataset and its characteristics.

Clean and preprocess the data.

Handle missing values.

Encode categorical features.

Select useful input features.

Split the data into training and testing sets.

Train regression models.

Evaluate model performance.

Analyze important features.

Generate predictions.

Save results and model-related outputs.

🎯 Objectives

Primary Objectives

Predict restaurant aggregate ratings using machine learning.

Perform data cleaning and preprocessing.

Handle missing and inconsistent data.

Encode categorical variables.

Train regression algorithms.

Compare model performance.

Evaluate predictions using regression metrics.

Analyze features that contribute to model predictions.

Secondary Objectives

Build a reproducible ML workflow.

Organize source code and generated outputs.

Maintain a clean project structure.

Provide model auditing utilities.

Document the complete project for future development.

📊 Dataset

The project uses a restaurant dataset containing information about
restaurants, locations, cuisines, pricing, services, customer votes, and
ratings.

The main dataset file is:

restaurants.csv

Dataset Attributes

Feature                  Description

Restaurant ID          Unique restaurant identifier
Restaurant Name        Restaurant name
Country Code           Country identifier
City                   Restaurant city
Address                Restaurant address
Locality               Local area or neighborhood
Locality Verbose       Detailed locality information
Longitude              Geographic longitude
Latitude               Geographic latitude
Cuisines               Cuisine or cuisines offered
Average Cost for two   Average cost for two people
Currency               Restaurant currency
Has Table booking      Whether table booking is available
Has Online delivery    Whether online delivery is available
Is delivering now      Whether the restaurant is currently delivering
Switch to order menu   Online ordering/menu availability
Price range            Restaurant price category
Aggregate rating       Overall restaurant rating and prediction target
Rating color           Rating color category
Rating text            Text representation of rating
Votes                  Number of customer votes

🎯 Target Variable

The target variable is:

Aggregate rating

The problem can therefore be represented as:

Input Restaurant Features
          ↓
Machine Learning Regression Model
          ↓
Predicted Aggregate Rating

🔄 Machine Learning Workflow

1. Data Loading

The dataset is loaded using Pandas.

import pandas as pd

df = pd.read_csv("restaurants.csv")

2. Data Inspection

The project examines:

Dataset shape

Data types

Missing values

Duplicate records

Numerical distributions

Categorical values

3. Data Cleaning

The preprocessing stage handles:

Missing values

Invalid or inconsistent values

Unnecessary columns

Data type conversion

Duplicate records where appropriate

4. Missing Value Handling

Numerical and categorical features are processed appropriately.

Typical strategies include:

Median imputation for numerical variables

Most-frequent-value imputation for categorical variables

5. Categorical Encoding

Categorical features are transformed into numerical representations
suitable for machine learning.

Possible techniques include:

One-Hot Encoding

Label Encoding

Binary encoding

6. Feature Engineering

Useful features are prepared from restaurant information, including:

Pricing information

Customer vote counts

Location information

Cuisine information

Service availability

Restaurant characteristics

7. Train/Test Split

The dataset is divided into:

Training Set → Used for learning
Testing Set  → Used for evaluating unseen data

This helps measure how well the trained model generalizes to unseen
restaurant records.

🤖 Models

The project uses regression algorithms to predict restaurant ratings.

Linear Regression

A simple baseline regression algorithm that models a linear relationship
between input features and the target.

Advantages - Simple - Fast - Easy to interpret - Useful as a
baseline

Limitations - Assumes linear relationships - May not capture complex
interactions

Decision Tree Regression

A tree-based algorithm that can model nonlinear relationships.

Advantages - Handles nonlinear patterns - Captures feature
interactions - Easy to visualize and understand

Limitations - Can overfit - Sensitive to training data

Random Forest Regression

An ensemble of decision trees that can capture complex relationships and
provide feature importance.

Advantages - Handles nonlinear relationships - Captures feature
interactions - More robust than a single tree - Supports feature
importance analysis

Limitations - More computationally expensive - Less interpretable
than simple linear regression - May require hyperparameter tuning

The exact algorithms executed are determined by the current
implementation in main.py.

📏 Model Evaluation

The project evaluates regression performance using standard metrics.

Mean Absolute Error --- MAE

Measures the average absolute difference between actual and predicted
ratings.

Lower MAE = Better performance

Mean Squared Error --- MSE

Measures the average squared prediction error.

Lower MSE = Better performance

Root Mean Squared Error --- RMSE

The square root of MSE.

Lower RMSE = Better performance

RMSE is expressed in the same unit as the target variable.

R² Score

Measures how much variation in restaurant ratings is explained by the
model.

Higher R² = Better explanatory performance

🔍 Feature Analysis

Feature analysis is used to understand which restaurant attributes
contribute to model predictions.

Potentially important factors include:

Customer votes

Average cost for two

Price range

City/location

Cuisine information

Table booking availability

Online delivery availability

Other processed restaurant attributes

For tree-based models, feature importance can be extracted from the
trained model.

Generated feature-analysis files and visualizations are stored under
outputs/.

📈 Exploratory Data Analysis

The project can analyze:

Rating Distribution

Understand the distribution of restaurant ratings.

Price Distribution

Study restaurant pricing patterns.

Votes vs Rating

Explore the relationship between customer engagement and ratings.

Location Analysis

Compare restaurant characteristics across locations.

Service Analysis

Study relationships between ratings and service availability.

Generated figures are stored in the outputs/ directory.

📁 Project Structure

Cognifyz_Task1_Restaurant_Rating/
│
├── data/
│   └── Dataset and supporting data files
│
├── notebooks/
│   └── Jupyter notebooks / experiments
│
├── outputs/
│   ├── figures/
│   │   └── Generated plots and visualizations
│   │
│   └── results/
│       └── Generated model results and predictions
│
├── src/
│   └── Source code for preprocessing and ML
│
├── venv/
│   └── Local Python virtual environment
│
├── .gitignore
│
├── main.py
│   └── Main machine learning pipeline
│
├── model_audit.py
│   └── Model validation and auditing utilities
│
├── restaurants.csv
│   └── Restaurant dataset
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation

Important: Do not upload the venv/ folder to GitHub. It is a
local Python environment and should be excluded through .gitignore.

🛠️ Technologies Used

Programming

Python 3.x

Data Analysis

Pandas

NumPy

Machine Learning

Scikit-learn

Visualization

Matplotlib

Seaborn

Model Persistence

Joblib

Development

Visual Studio Code

Jupyter Notebook

Version Control

Git

GitHub

📦 Requirements

Dependencies are listed in:

requirements.txt

Typical packages include:

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter

Use the exact versions in requirements.txt when reproducing the
project.

💻 Installation

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/Cognifyz_Task1_Restaurant_Rating_Prediction.git

2. Navigate to the Project

cd Cognifyz_Task1_Restaurant_Rating_Prediction

3. Create a Virtual Environment

python -m venv venv

4. Activate the Environment

Windows

venv\Scriptsctivate

Linux/macOS

source venv/bin/activate

5. Install Dependencies

pip install -r requirements.txt

▶️ How to Run

Activate the virtual environment and execute:

python main.py

The pipeline performs:

Load Dataset
      ↓
Inspect Dataset
      ↓
Clean Data
      ↓
Preprocess Features
      ↓
Split Dataset
      ↓
Train Regression Models
      ↓
Evaluate Models
      ↓
Generate Predictions
      ↓
Generate Visualizations
      ↓
Save Results

🔎 Model Audit

The repository contains:

model_audit.py

Run it with:

python model_audit.py

The audit utility can be used to inspect model-related artifacts and
validate the generated machine learning outputs.

📂 Outputs

Generated files are stored under:

outputs/

Typical structure:

outputs/
│
├── figures/
│   ├── *.png
│   └── *.jpg
│
└── results/
    ├── *.csv
    └── generated result files

These outputs can contain:

Model comparison results

Predictions

Feature analysis

Data visualizations

Evaluation metrics

📊 Results

The final metrics should always be taken from the actual output
generated by main.py.

Model                   MAE            MSE           RMSE             R²

Linear        See generated  See generated  See generated  See generated
Regression          results        results        results        results

Decision      See generated  See generated  See generated  See generated
Tree                results        results        results        results
Regression

No manually estimated performance values are included. Replace the
table values with the actual metrics produced by your final run if you
want the README to display them directly.

🧪 Reproducibility

To reproduce the project:

Clone the repository.

Create a new virtual environment.

Install the dependencies from requirements.txt.

Place the required dataset in the expected location.

Run python main.py.

Review the generated files in outputs/.

Keeping dependency versions fixed helps reduce differences between
development environments.

⚠️ Challenges and Limitations

1. Subjective Ratings

Restaurant ratings are influenced by individual customer experiences.

2. Dataset Limitations

The model can only learn from the information available in the dataset.

3. Missing Information

Factors such as food quality, ambience, service quality, and detailed
customer sentiment may not be represented.

4. Geographic Variation

Restaurant behavior and rating patterns can vary across cities and
countries.

5. Popularity Bias

Restaurants with many votes may have different patterns from restaurants
with few votes.

6. Feature Representation

Some categorical information may be difficult to represent completely
using standard encoding techniques.

7. Generalization

A model trained on this dataset may not perform identically on
restaurants from another region, platform, or time period.

🔮 Future Improvements

Machine Learning

Hyperparameter tuning

Cross-validation

Gradient Boosting

XGBoost

LightGBM

CatBoost

Ensemble learning

Feature Engineering

Better cuisine representation

Geographic distance features

Restaurant popularity indicators

Price normalization

More detailed service features

Additional temporal information

NLP

If customer reviews become available:

Sentiment analysis

Review classification

Keyword extraction

Review-based rating prediction

Explainable AI

Possible additions:

SHAP

LIME

Individual prediction explanations

Feature contribution analysis

Deployment

The model could be deployed through:

Streamlit

Flask

FastAPI

Django REST Framework

Cloud platforms

🎓 Learning Outcomes

This project provided hands-on practice with:

Python

Pandas

NumPy

Exploratory Data Analysis

Data cleaning

Missing value handling

Categorical encoding

Feature engineering

Regression algorithms

Train/test splitting

Model evaluation

Feature importance

Model persistence

Machine learning pipelines

Project organization

Git and GitHub

Reproducible ML workflows

📋 Internship Task Mapping

Requirement              Implementation

Restaurant dataset       restaurants.csv
Data inspection          Implemented
Data preprocessing       Implemented
Missing value handling   Implemented
Feature preparation      Implemented
Categorical encoding     Implemented where required
Regression problem       Aggregate Rating prediction
Train/test split         Implemented
Regression algorithms    Implemented in the project
Model evaluation         Regression metrics
Feature analysis         Feature importance / analysis
Results                  outputs/
Reproducibility          requirements.txt and documented workflow

🚀 Future Deployment Architecture

A production-oriented version could follow:

                    User
                     │
                     ▼
              Web Application
                     │
                     ▼
                  REST API
                     │
                     ▼
          Preprocessing Pipeline
                     │
                     ▼
             Trained ML Model
                     │
                     ▼
        Predicted Restaurant Rating

🔐 Data & Security Notes

Do not commit sensitive information to GitHub.

Never upload:

Passwords

API keys

Access tokens

Private credentials

.env files

Sensitive datasets

If the dataset is obtained from a third party, verify its licensing and
redistribution requirements before publishing the raw dataset.

🧹 Recommended .gitignore

# Virtual environments
venv/
.venv/
env/

# Python
__pycache__/
*.py[cod]
*.pyo

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
.env.*

# IDE
.vscode/
.idea/

# Operating system
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.log

If you intentionally want to publish trained .joblib or .pkl models,
do not add those extensions to .gitignore.

📌 GitHub Information

Repository Name

Repository Description

Machine learning project for predicting restaurant aggregate ratings using regression, data preprocessing, feature engineering, and model evaluation.

Suggested GitHub Topics

python
machine-learning
machine-learning-project
restaurant-rating-prediction
regression
scikit-learn
pandas
numpy
data-science
feature-engineering

👨‍💻 Author

Jibin

B.Tech Computer Science Engineering Student

Interests:

Machine Learning

Python

Cybersecurity

Software Development

Data Science

Developed as part of the Cognifyz Technologies Machine Learning
Internship.


🙏 Acknowledgement

Thanks to Cognifyz Technologies for providing the opportunity to
work on practical machine learning tasks and gain hands-on experience
in:

Data preprocessing

Exploratory data analysis

Machine learning

Model evaluation

Feature engineering

Model development

Project documentation

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on
GitHub.

Built with Python 🐍 | Machine Learning 🤖 | Data Science 📊
