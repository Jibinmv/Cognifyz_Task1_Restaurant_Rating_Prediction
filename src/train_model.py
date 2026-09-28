

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


def train_linear_regression(preprocessor, X_train, y_train):
    """
    Train a Linear Regression model with the preprocessing pipeline.
    
    Linear Regression finds the best-fitting linear relationship between
    the input features and the target variable. It minimizes the sum of
    squared differences between predicted and actual values.
    
    Parameters:
        preprocessor: A scikit-learn ColumnTransformer (unfitted or fitted).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
    
    Returns:
        sklearn.pipeline.Pipeline: A fitted pipeline (preprocessor + model).
    """
    # Create a full pipeline: preprocessing + linear regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    # Fit the pipeline on training data
    # The preprocessor is fitted here (learns encodings, scaling parameters)
    # and the model is trained on the transformed features
    pipeline.fit(X_train, y_train)
    
    print("Linear Regression model trained successfully.")
    return pipeline


def train_decision_tree(preprocessor, X_train, y_train, random_state=42):
    """
    Train a Decision Tree Regressor with the preprocessing pipeline.
    
    Decision Trees learn a set of if-then rules from the data to make
    predictions. They split the data recursively based on feature values
    to minimize prediction error in each resulting group.
    
    Parameters:
        preprocessor: A scikit-learn ColumnTransformer (unfitted or fitted).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: A fitted pipeline (preprocessor + model).
    """
    # Create a full pipeline: preprocessing + decision tree
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', DecisionTreeRegressor(random_state=random_state))
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    print("Decision Tree Regressor model trained successfully.")
    return pipeline


def predict(pipeline, X):
    """
    Use a trained pipeline to make predictions on new data.
    
    Parameters:
        pipeline: A fitted scikit-learn Pipeline.
        X (pd.DataFrame): Features to predict on.
    
    Returns:
        np.ndarray: Predicted aggregate ratings.
    """
    return pipeline.predict(X)


def predict_rating(pipeline, restaurant_data):
    """
    Predict the Aggregate rating for a single restaurant or a batch.
    
    This is a reusable prediction function that accepts restaurant feature
    values and returns the predicted rating using the trained pipeline.
    
    The pipeline internally handles preprocessing (scaling, encoding)
    so raw feature values can be passed directly.
    
    Parameters:
        pipeline: A fitted scikit-learn Pipeline (preprocessor + model).
        restaurant_data (dict or pd.DataFrame): Restaurant feature values.
            If dict, keys should be feature names and values should be
            the corresponding feature values for one restaurant.
            If DataFrame, each row is a restaurant.
    
    Returns:
        float or np.ndarray: Predicted aggregate rating(s).
    
    Example:
        restaurant = {
            'Country Code': 1,
            'City': 'New Delhi',
            'Longitude': 77.2,
            'Latitude': 28.6,
            'Cuisines': 'North Indian',
            'Average Cost for two': 500,
            'Currency': 'Indian Rupees(Rs.)',
            'Has Table booking': 'No',
            'Has Online delivery': 'Yes',
            'Is delivering now': 'No',
            'Price range': 2,
            'Votes': 100,
        }
        predicted_rating = predict_rating(pipeline, restaurant)
    """
    # Convert dict to DataFrame if necessary
    if isinstance(restaurant_data, dict):
        restaurant_data = pd.DataFrame([restaurant_data])
    
    prediction = pipeline.predict(restaurant_data)
    
    # If single restaurant, return a scalar
    if len(prediction) == 1:
        return float(prediction[0])
    return prediction
