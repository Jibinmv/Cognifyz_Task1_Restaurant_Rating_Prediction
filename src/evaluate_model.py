

import pandas as pd
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




def evaluate_model(y_true, y_pred, model_name="Model"):
  
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
    }
    
    print(f"\n{model_name} Evaluation:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def create_comparison_table(metrics_list, save_path=None):
    
    comparison_df = pd.DataFrame(metrics_list)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison_df.to_csv(save_path, index=False)
        print(f"\nComparison table saved to: {save_path}")
    
    return comparison_df



def plot_rating_distribution(y, save_path=None):
    """
    Plot the distribution of the target variable (Aggregate rating).
    
    Parameters:
        y (array-like): Aggregate rating values.
        save_path (str, optional): Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(y, bins=30, kde=True, color='steelblue', ax=ax)
    ax.set_title('Distribution of Aggregate Rating', fontsize=16, fontweight='bold')
    ax.set_xlabel('Aggregate Rating', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.axvline(x=y.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {y.mean():.2f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rating distribution plot saved to: {save_path}")
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path=None):
    """
    Create a scatter plot comparing actual vs predicted ratings.
    
    A perfect model would have all points on the diagonal line (y=x).
    Points above the line are over-predicted, below are under-predicted.
    
    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model for the title.
        save_path (str, optional): Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color='steelblue',
               edgecolors='none')
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')
    
    ax.set_title(f'Actual vs Predicted - {model_name}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Aggregate Rating', fontsize=13)
    ax.set_ylabel('Predicted Aggregate Rating', fontsize=13)
    ax.legend(fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Actual vs Predicted plot saved to: {save_path}")
    plt.close(fig)


def plot_model_comparison(comparison_df, save_path=None):
    """
    Create a bar chart comparing model performance metrics.
    
    Parameters:
        comparison_df (pd.DataFrame): Comparison table from create_comparison_table().
        save_path (str, optional): Path to save the figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']
    colors = ['#2196F3', '#FF9800']
    labels = comparison_df['Model'].tolist()
    
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].tolist()
        bars = axes[i].bar(labels, values, color=colors[:len(labels)],
                           edgecolor='black', linewidth=0.5)
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Score', fontsize=12)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[i].tick_params(axis='x', rotation=15)
    
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    plt.close(fig)




def extract_feature_importance(pipeline, feature_names, save_csv_path=None,
                               save_plot_path=None, top_n=15):
    """
    Extract and visualize feature importances from a Decision Tree model.
    
    Feature importance in a Decision Tree measures how much each feature
    contributes to reducing prediction error across all splits in the tree.
    A higher importance means the feature is used more often and/or causes
    larger reductions in error.
    
    NOTE: Feature importance reflects model-derived importance, which
    indicates how the model uses features for prediction. It does NOT
    prove causation between a feature and the target variable.
    
    Parameters:
        pipeline: A fitted Pipeline containing a Decision Tree model.
        feature_names (list): Transformed feature names from the preprocessor.
        save_csv_path (str, optional): Path to save importance table as CSV.
        save_plot_path (str, optional): Path to save importance plot.
        top_n (int): Number of top features to show in the plot.
    
    Returns:
        pd.DataFrame: Sorted feature importance table.
    """
    # Extract the model from the pipeline
    model = pipeline.named_steps['model']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame with feature names and their importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Decision Tree - Model-Derived)")
    print("=" * 60)
    print("\nNote: This shows how the model uses features for prediction.")
    print("It does NOT prove causation.\n")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Save to CSV
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        importance_df.to_csv(save_csv_path, index=False)
        print(f"\nFeature importance table saved to: {save_csv_path}")
    
    # Plot top features
    if save_plot_path:
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(
            range(len(top_features)),
            top_features['Importance'].values,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Importance', fontsize=13)
        ax.set_title(f'Top {top_n} Feature Importances (Decision Tree)',
                     fontsize=16, fontweight='bold')
    
        for bar, val in zip(bars, top_features['Importance'].values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        fig.savefig(save_plot_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_plot_path}")
        plt.close(fig)
    
    return importance_df
