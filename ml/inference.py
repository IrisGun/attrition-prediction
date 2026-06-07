import pandas as pd
import numpy as np
import pickle
import os
import json

def run_inference(snapshot_date='2026-03-01'):
    """
    Load the best model and current snapshot data, generate predictions.
    """
    # Load the best model
    model_path = 'ml/models/best_model.pkl'
    if not os.path.exists(model_path):
        print("Best model not found. Run train_eval.py first.")
        return
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Load current snapshot data
    features_path = f'ml/data/processed/features_{snapshot_date}.csv'
    if not os.path.exists(features_path):
        print(f"Processed features for {snapshot_date} not found. Run feature_engineering.py first.")
        return
        
    df = pd.read_csv(features_path)
    
    # Define features
    features = ['tenure_days', 'tenure_months', 'days_worked_last_30', 'ewa_count_60', 'ewa_sum_60', 'ewa_avg_60', 'company_cat', 'dept_cat', 'city_cat']
    X = df[features]
    
    # Generate predictions (probability of attrition)
    # Use predict_proba if available, otherwise predict
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)
        
    df['risk_score'] = (y_prob * 100).astype(int)
    
    # Identify high-risk employees
    high_risk_threshold = 70
    high_risk_emps = df[df['risk_score'] >= high_risk_threshold].copy()
    
    # Prepare results for frontend
    results = {
        'total_employees': int(len(df)),
        'high_risk_count': int(len(high_risk_emps)),
        'avg_risk_score': int(df['risk_score'].mean()),
        'risk_distribution': [
            {'name': 'High Risk', 'value': int(len(df[df['risk_score'] >= 70])), 'color': '#ef4444'},
            {'name': 'Medium Risk', 'value': int(len(df[(df['risk_score'] >= 40) & (df['risk_score'] < 70)])), 'color': '#f59e0b'},
            {'name': 'Low Risk', 'value': int(len(df[df['risk_score'] < 40])), 'color': '#10b981'},
        ],
        'top_high_risk': high_risk_emps.sort_values(by='risk_score', ascending=False).head(5)[['emp_id', 'company', 'dept', 'tenure_months', 'risk_score']].to_dict(orient='records')
    }
    
    # Save results to artifacts
    os.makedirs('ml/artifacts', exist_ok=True)
    with open('ml/artifacts/predictions.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Inference complete for {snapshot_date}. Results saved to ml/artifacts/predictions.json")
    return results

if __name__ == "__main__":
    run_inference()
