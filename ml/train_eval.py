import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from models import AttritionModel, compare_models

def train_and_evaluate():
    """
    Load processed features, split into train/test, and compare models.
    """
    # Load processed features (using multiple snapshots for training)
    snapshots = ['2025-01-01', '2025-06-01', '2025-12-01']
    all_features = []
    for snapshot in snapshots:
        features_path = f'ml/data/processed/features_{snapshot}.csv'
        if os.path.exists(features_path):
            all_features.append(pd.read_csv(features_path))
            
    if not all_features:
        print("No processed features found. Run feature_engineering.py first.")
        return
        
    df = pd.concat(all_features, ignore_index=True)
    
    # Define features and target
    features = ['tenure_days', 'tenure_months', 'days_worked_last_30', 'ewa_count_60', 'ewa_sum_60', 'ewa_avg_60', 'company_cat', 'dept_cat', 'city_cat']
    target = 'resigned_next_3m'
    
    X = df[features]
    y = df[target]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare models and pick the best one
    best_model_type, results = compare_models(X_train, y_train, X_test, y_test)
    
    # Save final metrics
    os.makedirs('ml/artifacts', exist_ok=True)
    with open('ml/artifacts/metrics.json', 'w') as f:
        json.dump({
            'best_model_type': best_model_type,
            'metrics': results[best_model_type]
        }, f, indent=4)
        
    print(f"Training and evaluation complete. Best model: {best_model_type}")
    return best_model_type, results

if __name__ == "__main__":
    train_and_evaluate()
