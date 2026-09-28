# 🍽️ Restaurant Rating Prediction

## 🎯 Objective

Build a machine learning regression system that predicts the **Aggregate rating** of a restaurant based on available restaurant features.

## 📋 Problem Statement

Given a dataset of 9,551 restaurants with various features (location, cuisine type, cost, online delivery availability, etc.), the goal is to train regression models to predict how a restaurant will be rated. This project compares **Linear Regression** and **Decision Tree Regression** to determine which model better captures the patterns in the data.

## 📊 Dataset Description

- **Source:** `data/restaurants.csv`
- **Rows:** 9,551
- **Columns:** 21

### All Columns

| Column | Type | Description |
|--------|------|-------------|
| Restaurant ID | int | Unique identifier |
| Restaurant Name | str | Name of the restaurant |
| Country Code | int | Country identifier (15 unique countries) |
| City | str | City where the restaurant is located (141 cities) |
| Address | str | Full address |
| Locality | str | Locality name |
| Locality Verbose | str | Detailed locality with city |
| Longitude | float | Geographic longitude |
| Latitude | float | Geographic latitude |
| Cuisines | str | Cuisine types offered (1,825 unique combinations) |
| Average Cost for two | int | Average meal cost for two people |
| Currency | str | Local currency (12 currencies) |
| Has Table booking | str | Yes/No - table reservation available |
| Has Online delivery | str | Yes/No - online delivery available |
| Is delivering now | str | Yes/No - currently delivering |
| Switch to order menu | str | Yes/No (constant "No" for all rows) |
| Price range | int | 1 to 4 scale |
| **Aggregate rating** | **float** | **Target variable (0.0 to 4.9)** |
| Rating color | str | Color code derived from rating |
| Rating text | str | Text label derived from rating |
| Votes | int | Number of votes received |

### Target Variable

**Aggregate rating** — continuous value ranging from 0.0 to 4.9.

- 2,148 restaurants (22.5%) have an Aggregate rating of 0.
- These are **"Not rated"** restaurants (Rating text = "Not rated"), meaning they haven't received enough user ratings. The zero is a placeholder, not an actual score.
- These records are retained in the dataset as valid data points.

## ⚙️ Data Preprocessing

### Missing Values
- **Cuisines:** 9 missing values → filled with `'Unknown'`
- All other columns: no missing values

### Duplicate Rows
- **0 duplicate rows** found

### Feature Selection

**Selected Features (12):**

| Feature | Type | Rationale |
|---------|------|-----------|
| Longitude | Numerical | Geographic signal |
| Latitude | Numerical | Geographic signal |
| Average Cost for two | Numerical | Price-quality indicator |
| Votes | Numerical | Popularity signal |
| Price range | Numerical | Categorical price tier (1-4) |
| Country Code | Categorical | Country-level differences |
| City | Categorical | City-level rating patterns |
| Cuisines | Categorical | Cuisine type affects ratings |
| Currency | Categorical | Regional proxy |
| Has Table booking | Categorical | Service quality signal |
| Has Online delivery | Categorical | Service availability |
| Is delivering now | Categorical | Active delivery status |

**Excluded Features:**

| Feature | Reason |
|---------|--------|
| Restaurant ID | Identifier — no predictive meaning |
| Restaurant Name | High cardinality (7,446 unique) — acts as identifier |
| Address | High cardinality (8,918 unique) — too specific |
| Locality | High cardinality (1,208 unique) — too many dummy variables |
| Locality Verbose | High cardinality (1,265 unique) — redundant with Locality |
| Rating color | **DATA LEAKAGE** — directly derived from Aggregate rating |
| Rating text | **DATA LEAKAGE** — directly derived from Aggregate rating |
| Switch to order menu | Constant value (all "No") — zero information |

### Pipeline Architecture

Built using scikit-learn `ColumnTransformer` and `Pipeline`:

- **Numerical:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical:** `SimpleImputer(strategy='constant', fill_value='Unknown')` → `OneHotEncoder(handle_unknown='ignore')`

### Train-Test Split

- **Training:** 80% (7,640 samples)
- **Testing:** 20% (1,911 samples)
- **Random state:** 42

## 🤖 Models Used

1. **Linear Regression** — Finds the best-fitting linear relationship between features and the target.
2. **Decision Tree Regressor** — Learns if-then rules by recursively splitting data to minimize prediction error.

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — average absolute prediction error |
| MSE | Mean Squared Error — average squared error (penalizes large errors) |
| RMSE | Root MSE — same units as target, sensitive to outliers |
| R² | Coefficient of Determination — proportion of variance explained (1.0 = perfect) |

## 📊 Results

| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|-----|
| Linear Regression | 0.9746 | 1.4529 | 1.2053 | 0.3617 |
| Decision Tree | 0.2574 | 0.1586 | 0.3982 | 0.9303 |

**Key Findings:**

- The **Decision Tree** significantly outperforms Linear Regression across all metrics.
- Decision Tree achieves **R² = 0.9303**, explaining ~93% of the variance in ratings.
- Linear Regression achieves **R² = 0.3617**, explaining only ~36% of the variance, suggesting the relationship between features and ratings is **non-linear**.

## 🔍 Feature Importance (Decision Tree — Model-Derived)

> **Note:** Feature importance reflects how the model uses features for prediction. It does **not** prove causation.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Votes | 0.9445 |
| 2 | Longitude | 0.0173 |
| 3 | Latitude | 0.0107 |
| 4 | Average Cost for two | 0.0054 |
| 5 | Country Code 30 | 0.0008 |

**Votes** is by far the most important feature, which makes intuitive sense: restaurants with more votes tend to be more popular and have more stable ratings.

## 🚀 How to Install

```bash
# Clone or download the project
cd Cognifyz_Task1_Restaurant_Rating

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

### Run the complete pipeline:

```bash
python main.py
```

This executes the full workflow: data loading → exploration → cleaning → preprocessing → model training → evaluation → saving results and visualizations.

### Run the Jupyter notebook:

```bash
jupyter notebook notebooks/task1_rating_prediction.ipynb
```

## 📁 Project Structure

```
Cognifyz_Task1_Restaurant_Rating/
│
├── data/
│   └── restaurants.csv              # Original dataset (9,551 rows × 21 columns)
│
├── notebooks/
│   └── task1_rating_prediction.ipynb # Interactive notebook with full analysis
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── preprocessing.py              # Data loading, cleaning, feature selection, pipeline
│   ├── train_model.py                # Model training and prediction functions
│   └── evaluate_model.py             # Evaluation metrics, visualizations, feature importance
│
├── outputs/
│   ├── figures/
│   │   ├── rating_distribution.png   # Target variable distribution
│   │   ├── actual_vs_predicted_lr.png # Linear Regression predictions
│   │   ├── actual_vs_predicted_dt.png # Decision Tree predictions
│   │   ├── model_comparison.png      # Side-by-side metric comparison
│   │   └── feature_importance.png    # Top 15 feature importances
│   └── results/
│       ├── model_comparison.csv      # Model metrics comparison table
│       └── feature_importance.csv    # Full feature importance rankings
│
├── main.py                           # Main entry point — runs complete pipeline
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🏁 Conclusion

This project demonstrates a complete machine learning regression pipeline for predicting restaurant ratings. The **Decision Tree Regressor** substantially outperformed **Linear Regression** (R² of 0.9303 vs 0.3617), indicating that the relationships between restaurant features and ratings are non-linear. The most influential feature was **Votes**, dominating the Decision Tree's importance scores with 94.5% of total importance. Geographic features (Longitude, Latitude) and cost-related features also contributed, while specific cuisine types showed minor but measurable effects.
=======
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

git clone https://github.com/Jibinmv/Cognifyz_Task1_Restaurant_Rating_Prediction.git
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
