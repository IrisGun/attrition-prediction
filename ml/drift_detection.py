import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import json

def detect_drift(training_snapshot='2025-12-01', current_snapshot='2026-03-01'):
    """
    Detect data drift by comparing feature distributions between training and current data.
    """
    # Load training and current data
    training_path = f'ml/data/processed/features_{training_snapshot}.csv'
    current_path = f'ml/data/processed/features_{current_snapshot}.csv'
    
    if not os.path.exists(training_path) or not os.path.exists(current_path):
        print("Processed features for training or current snapshot not found.")
        return False
        
    training_df = pd.read_csv(training_path)
    current_df = pd.read_csv(current_path)
    
    # Define features to check for drift
    features = ['tenure_days', 'days_worked_last_30', 'ewa_count_60', 'ewa_sum_60', 'ewa_avg_60']
    
    drift_results = {}
    drift_detected = False
    
    for feature in features:
        # Perform Kolmogorov-Smirnov test for distribution comparison
        ks_stat, p_value = ks_2samp(training_df[feature], current_df[feature])
        
        # If p-value is small, distributions are significantly different (drift)
        is_drift = p_value < 0.05
        drift_results[feature] = {
            'ks_stat': float(ks_stat),
            'p_value': float(p_value),
            'is_drift': bool(is_drift)
        }
        if is_drift:
            drift_detected = True
            
    # Save drift results
    os.makedirs('ml/artifacts', exist_ok=True)
    with open('ml/artifacts/drift_results.json', 'w') as f:
        json.dump({
            'drift_detected': drift_detected,
            'feature_drift': drift_results
        }, f, indent=4)
        
    if drift_detected:
        print("Data drift detected! Re-training recommended.")
    else:
        print("No significant data drift detected.")
        
    return drift_detected

if __name__ == "__main__":
    detect_drift()
