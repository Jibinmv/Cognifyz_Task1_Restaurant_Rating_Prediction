
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path so we can import src modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from src.preprocessing import (
    load_data,
    explore_data,
    clean_data,
    get_feature_selection_report,
    build_preprocessing_pipeline,
    split_data,
    get_feature_names_from_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.train_model import (
    train_linear_regression,
    train_decision_tree,
    predict_rating,
)
from src.evaluate_model import (
    evaluate_model,
    create_comparison_table,
    plot_rating_distribution,
    plot_actual_vs_predicted,
    plot_model_comparison,
    extract_feature_importance,
)


def main():
    """Execute the complete machine learning pipeline."""
    
    print("=" * 70)
    print("  COGNIFYZ TASK 1: RESTAURANT RATING PREDICTION")
    print("  Machine Learning Internship Project")
    print("=" * 70)
    
    
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    
    data_path = os.path.join(project_root, 'data', 'restaurants.csv')
    
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure 'restaurants.csv' is in the 'data/' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    
    print("\n" + "=" * 70)
    print("STEP 2: DATA EXPLORATION")
    print("=" * 70)
    
    summary = explore_data(df)
    
   
    print("\n" + "=" * 70)
    print("STEP 3: DATA CLEANING")
    print("=" * 70)
    
    df_clean = clean_data(df)
    
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE SELECTION")
    print("=" * 70)
    
    feature_report = get_feature_selection_report()
    
    
    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESSING PIPELINE")
    print("=" * 70)
    
    preprocessor = build_preprocessing_pipeline()
    
   
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    print("\n" + "=" * 70)
    print("STEP 7: TRAINING LINEAR REGRESSION")
    print("=" * 70)
    
    lr_pipeline = train_linear_regression(preprocessor, X_train, y_train)
    lr_predictions = lr_pipeline.predict(X_test)
    

    print("\n" + "=" * 70)
    print("STEP 8: TRAINING DECISION TREE REGRESSION")
    print("=" * 70)
    
    # Build a fresh preprocessor for the Decision Tree
    # (each pipeline needs its own preprocessor instance)
    preprocessor_dt = build_preprocessing_pipeline()
    dt_pipeline = train_decision_tree(preprocessor_dt, X_train, y_train)
    dt_predictions = dt_pipeline.predict(X_test)
    
    
    print("\n" + "=" * 70)
    print("STEP 9: MODEL EVALUATION")
    print("=" * 70)
    
    lr_metrics = evaluate_model(y_test, lr_predictions, "Linear Regression")
    dt_metrics = evaluate_model(y_test, dt_predictions, "Decision Tree")
    
   
    print("\n" + "=" * 70)
    print("STEP 10: SAVING RESULTS AND VISUALIZATIONS")
    print("=" * 70)
    
    # Create output directories
    figures_dir = os.path.join(project_root, 'outputs', 'figures')
    results_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model comparison table
    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df = create_comparison_table(
        [lr_metrics, dt_metrics],
        save_path=comparison_path
    )
    
    # Plot 1: Rating distribution
    plot_rating_distribution(
        df_clean['Aggregate rating'],
        save_path=os.path.join(figures_dir, 'rating_distribution.png')
    )
    
    # Plot 2: Actual vs Predicted - Linear Regression
    plot_actual_vs_predicted(
        y_test, lr_predictions, 'Linear Regression',
        save_path=os.path.join(figures_dir, 'actual_vs_predicted_lr.png')
    )
    
    # Plot 3: Actual vs Predicted - Decision Tree
    plot_actual_vs_predicted(
        y_test, dt_predictions, 'Decision Tree',
        save_path=os.path.join(figures_dir, 'actual_vs_predicted_dt.png')
    )
    
    # Plot 4: Model performance comparison
    plot_model_comparison(
        comparison_df,
        save_path=os.path.join(figures_dir, 'model_comparison.png')
    )
    
    
    print("\n" + "=" * 70)
    print("STEP 11: FEATURE IMPORTANCE (DECISION TREE)")
    print("=" * 70)
    
    # Get transformed feature names from the Decision Tree's preprocessor
    feature_names = get_feature_names_from_preprocessor(
        dt_pipeline.named_steps['preprocessor']
    )
    
    importance_df = extract_feature_importance(
        dt_pipeline,
        feature_names,
        save_csv_path=os.path.join(results_dir, 'feature_importance.csv'),
        save_plot_path=os.path.join(figures_dir, 'feature_importance.png'),
        top_n=15
    )
    
  
    print("\n" + "=" * 70)
    print("STEP 12: SAMPLE PREDICTION")
    print("=" * 70)
    
    sample = X_test.iloc[0:1]
    actual_rating = y_test.iloc[0]
    
    predicted_lr = predict_rating(lr_pipeline, sample)
    predicted_dt = predict_rating(dt_pipeline, sample)
    
    print(f"\nSample Restaurant Features:")
    for col in sample.columns:
        print(f"  {col}: {sample[col].values[0]}")
    print(f"\nActual Aggregate Rating: {actual_rating}")
    print(f"Linear Regression Prediction: {predicted_lr:.4f}")
    print(f"Decision Tree Prediction:     {predicted_dt:.4f}")
    
 
    print("\n--- Prediction from dictionary input ---")
    sample_dict = {
        'Country Code': 1,
        'City': 'New Delhi',
        'Longitude': 77.2,
        'Latitude': 28.6,
        'Cuisines': 'North Indian, Chinese',
        'Average Cost for two': 500,
        'Currency': 'Indian Rupees(Rs.)',
        'Has Table booking': 'No',
        'Has Online delivery': 'Yes',
        'Is delivering now': 'No',
        'Price range': 2,
        'Votes': 100,
    }
    
    pred_dict_lr = predict_rating(lr_pipeline, sample_dict)
    pred_dict_dt = predict_rating(dt_pipeline, sample_dict)
    
    print(f"  Linear Regression Prediction: {pred_dict_lr:.4f}")
    print(f"  Decision Tree Prediction:     {pred_dict_dt:.4f}")
    
  
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"""
  Dataset:          {df.shape[0]} restaurants, {df.shape[1]} columns
  Target Variable:  Aggregate rating
  Zero Ratings:     {summary['zero_ratings']} (unrated restaurants)
  Missing Values:   Cuisines ({df['Cuisines'].isnull().sum()} rows) -> filled with 'Unknown'
  Features Used:    {len(NUMERICAL_FEATURES)} numerical + {len(CATEGORICAL_FEATURES)} categorical
  Train/Test Split: 80% / 20% (random_state=42)
  
  Model Performance (Test Set):
  {comparison_df.to_string(index=False)}
  
  Output Files:
    - outputs/results/model_comparison.csv
    - outputs/results/feature_importance.csv
    - outputs/figures/rating_distribution.png
    - outputs/figures/actual_vs_predicted_lr.png
    - outputs/figures/actual_vs_predicted_dt.png
    - outputs/figures/model_comparison.png
    - outputs/figures/feature_importance.png
""")
    print("=" * 70)
    print("  Pipeline completed successfully!")
    print("=" * 70)
    
    return {
        'lr_pipeline': lr_pipeline,
        'dt_pipeline': dt_pipeline,
        'comparison_df': comparison_df,
        'importance_df': importance_df,
    }


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
