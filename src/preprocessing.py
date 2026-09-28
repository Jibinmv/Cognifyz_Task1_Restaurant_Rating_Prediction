

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer



def load_data(filepath):
    """
    Load the restaurant dataset from a CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df




def explore_data(df):
    """
    Perform comprehensive data exploration and return a summary dictionary.
    
    This function prints and collects information about:
    - Shape, column names, data types
    - Missing values and duplicate rows
    - Descriptive statistics
    - Target variable distribution
    - Zero ratings analysis
    
    Parameters:
        df (pd.DataFrame): The dataset to explore.
    
    Returns:
        dict: A dictionary containing exploration results.
    """
    summary = {}
    
    # --- Shape ---
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    summary['shape'] = df.shape
    
    # --- Column names and data types ---
    print("Column Names and Data Types:")
    print("-" * 40)
    for col in df.columns:
        print(f"  {col:30s} -> {df[col].dtype}")
    summary['dtypes'] = df.dtypes.to_dict()
    
    # --- Missing values ---
    print("\nMissing Values:")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct.round(2)
    })
    # Only show columns with missing values or show all
    has_missing = missing_df[missing_df['Missing Count'] > 0]
    if len(has_missing) > 0:
        print(has_missing.to_string())
    else:
        print("  No missing values found.")
    summary['missing_values'] = missing.to_dict()
    
    # --- Duplicate rows ---
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count}")
    summary['duplicates'] = dup_count
    
    # --- Descriptive statistics for numerical columns ---
    print("\nDescriptive Statistics (Numerical Columns):")
    print("-" * 40)
    desc = df.describe()
    print(desc.to_string())
    summary['describe'] = desc
    
    # --- Unique values for key categorical columns ---
    print("\nUnique Values for Key Categorical Columns:")
    print("-" * 40)
    categorical_cols = ['Has Table booking', 'Has Online delivery',
                        'Is delivering now', 'Switch to order menu',
                        'Rating color', 'Rating text']
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].unique()
            print(f"  {col}: {unique_vals} ({len(unique_vals)} unique)")
    
    # --- Cardinality of high-cardinality columns ---
    print("\nCardinality of Text/Categorical Columns:")
    print("-" * 40)
    for col in ['Restaurant Name', 'City', 'Address', 'Locality',
                'Locality Verbose', 'Cuisines', 'Currency', 'Country Code']:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")
    
    # --- Target variable analysis ---
    print("\nTarget Variable: Aggregate rating")
    print("-" * 40)
    print(df['Aggregate rating'].describe().to_string())
    
    zero_count = (df['Aggregate rating'] == 0).sum()
    print(f"\nZero Ratings (Aggregate rating == 0): {zero_count}")
    print(f"  Percentage: {zero_count / len(df) * 100:.2f}%")
    summary['zero_ratings'] = zero_count
    
    # --- Investigate zero ratings ---
    zero_df = df[df['Aggregate rating'] == 0]
    print("\n  Analysis of Zero-Rated Restaurants:")
    if 'Rating text' in df.columns:
        print(f"    Rating text distribution:")
        for val, cnt in zero_df['Rating text'].value_counts().items():
            print(f"      {val}: {cnt}")
    if 'Votes' in df.columns:
        print(f"    Votes statistics:")
        print(f"      Mean: {zero_df['Votes'].mean():.2f}")
        print(f"      Max:  {zero_df['Votes'].max()}")
        print(f"      Votes == 0: {(zero_df['Votes'] == 0).sum()}")
    
    print("\n  INTERPRETATION: Restaurants with Aggregate rating = 0 are")
    print("  labeled 'Not rated' in the Rating text column. These are")
    print("  restaurants that have NOT been rated by users, not restaurants")
    print("  that received a score of zero. The zero is a placeholder for")
    print("  the absence of a rating.")
    
    summary['exploration_complete'] = True
    return summary




def clean_data(df):
    """
    Clean the dataset by handling missing values.
    
    Operations:
    - Fill missing 'Cuisines' values with 'Unknown'
    - Do NOT modify the original CSV file
    - Do NOT automatically remove zero-rated restaurants
    
    Parameters:
        df (pd.DataFrame): The raw dataset.
    
    Returns:
        pd.DataFrame: A cleaned copy of the dataset.
    """
    # Work on a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing Cuisines (9 rows)
    cuisines_missing = df_clean['Cuisines'].isnull().sum()
    if cuisines_missing > 0:
        df_clean['Cuisines'] = df_clean['Cuisines'].fillna('Unknown')
        print(f"Filled {cuisines_missing} missing Cuisines values with 'Unknown'")
    
    # Check for any other missing values
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after cleaning")
    else:
        print("No remaining missing values after cleaning.")
    
    # Note about zero ratings - keep them with explanation
    zero_count = (df_clean['Aggregate rating'] == 0).sum()
    print(f"\nZero-rated restaurants retained: {zero_count}")
    print("  These represent 'Not rated' restaurants. They are kept in the")
    print("  dataset because they are valid data points - the model should")
    print("  learn to predict low/zero ratings for unrated restaurants based")
    print("  on their features (e.g., low votes, no online delivery, etc.).")
    
    print(f"\nCleaned dataset shape: {df_clean.shape}")
    return df_clean




# --- Features to EXCLUDE and why ---
EXCLUDED_FEATURES = {
    'Restaurant ID': 'Unique identifier with no predictive meaning',
    'Restaurant Name': 'High cardinality (7446 unique) - acts as an identifier',
    'Address': 'High cardinality (8918 unique) - too specific, acts as identifier',
    'Locality': 'High cardinality (1208 unique) - would create too many dummy variables',
    'Locality Verbose': 'High cardinality (1265 unique) - similar to Locality',
    'Rating color': 'DATA LEAKAGE - directly derived from Aggregate rating',
    'Rating text': 'DATA LEAKAGE - directly derived from Aggregate rating',
    'Aggregate rating': 'This is the TARGET variable, not a feature',
    'Switch to order menu': 'Constant value (all "No") - provides no information',
}

# --- Features to USE ---
NUMERICAL_FEATURES = [
    'Longitude',
    'Latitude',
    'Average Cost for two',
    'Votes',
    'Price range',
]

CATEGORICAL_FEATURES = [
    'Country Code',       # 15 unique values - manageable for one-hot encoding
    'City',               # 141 unique values - kept because city affects ratings
    'Cuisines',           # Will be one-hot encoded (high cardinality but informative)
    'Currency',           # 12 unique values - proxy for country/region
    'Has Table booking',  # Binary: Yes/No
    'Has Online delivery',# Binary: Yes/No
    'Is delivering now',  # Binary: Yes/No
]

TARGET = 'Aggregate rating'


def get_feature_selection_report():
    """
    Print and return the feature selection decisions with explanations.
    
    Returns:
        dict: Feature selection details.
    """
    print("=" * 60)
    print("FEATURE SELECTION REPORT")
    print("=" * 60)
    
    print("\nEXCLUDED Features:")
    print("-" * 40)
    for feat, reason in EXCLUDED_FEATURES.items():
        print(f"  {feat:25s} -> {reason}")
    
    print(f"\nSELECTED Numerical Features ({len(NUMERICAL_FEATURES)}):")
    print("-" * 40)
    for feat in NUMERICAL_FEATURES:
        print(f"  - {feat}")
    
    print(f"\nSELECTED Categorical Features ({len(CATEGORICAL_FEATURES)}):")
    print("-" * 40)
    for feat in CATEGORICAL_FEATURES:
        print(f"  - {feat}")
    
    print(f"\nTARGET: {TARGET}")
    
    return {
        'numerical': NUMERICAL_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'excluded': EXCLUDED_FEATURES,
        'target': TARGET,
    }


def build_preprocessing_pipeline():
    """
    Build a scikit-learn ColumnTransformer preprocessing pipeline.
    
    Numerical features:
        - Impute missing values with the median
        - Scale using StandardScaler
    
    Categorical features:
        - Impute missing values with the constant 'Unknown'
        - One-hot encode with handle_unknown='ignore'
    
    Returns:
        sklearn.compose.ColumnTransformer: The preprocessing pipeline.
    """
    # Numerical pipeline: impute missing values then scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing values then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder='drop'  # Drop any columns not listed above
    )
    
    print("Preprocessing pipeline built successfully.")
    print(f"  Numerical features ({len(NUMERICAL_FEATURES)}): impute median + scale")
    print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): impute 'Unknown' + one-hot encode")
    
    return preprocessor



def split_data(df, target_col=TARGET, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The cleaned dataset.
        target_col (str): Name of the target column.
        test_size (float): Proportion for test set (default 0.2 = 20%).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Select only the features we want to use
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData Split (test_size={test_size}, random_state={random_state}):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")
    print(f"  Features:     {X_train.shape[1]} columns")
    
    return X_train, X_test, y_train, y_test


def get_feature_names_from_preprocessor(preprocessor):
    """
    Extract the transformed feature names from a fitted ColumnTransformer.
    
    This is important for interpreting feature importances after one-hot
    encoding creates many new columns from categorical features.
    
    Parameters:
        preprocessor (ColumnTransformer): A fitted ColumnTransformer.
    
    Returns:
        list: List of transformed feature names.
    """
    return preprocessor.get_feature_names_out().tolist()
