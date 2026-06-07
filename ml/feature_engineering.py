import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def process_features(snapshot_date='2026-03-01'):
    """
    Load raw data and create features for a specific snapshot date.
    Handles data variations by using robust feature creation.
    """
    # Load raw data
    emp_master = pd.read_csv('ml/data/raw/employee_master.csv')
    attendance = pd.read_csv('ml/data/raw/attendance.csv')
    ewa_usage = pd.read_csv('ml/data/raw/ewa_usage.csv')
    resignation = pd.read_csv('ml/data/raw/resignation.csv')
    
    # Convert dates
    snapshot_dt = pd.to_datetime(snapshot_date)
    emp_master['join_date'] = pd.to_datetime(emp_master['join_date'])
    attendance['date'] = pd.to_datetime(attendance['date'])
    ewa_usage['date'] = pd.to_datetime(ewa_usage['date'])
    resignation['resign_date'] = pd.to_datetime(resignation['resign_date'])
    
    # 1. Filter active employees at snapshot date
    # Active = Joined before snapshot AND (Resigned after snapshot OR Not resigned)
    resigned_before = resignation[resignation['resign_date'] <= snapshot_dt]['emp_id'].tolist()
    active_emps = emp_master[(emp_master['join_date'] <= snapshot_dt) & (~emp_master['emp_id'].isin(resigned_before))].copy()
    
    # 2. Tenure Feature
    active_emps['tenure_days'] = (snapshot_dt - active_emps['join_date']).dt.days
    active_emps['tenure_months'] = active_emps['tenure_days'] // 30
    
    # 3. Attendance Features (Last 30 days)
    last_30_days = attendance[(attendance['date'] > snapshot_dt - timedelta(days=30)) & (attendance['date'] <= snapshot_dt)]
    attendance_counts = last_30_days.groupby('emp_id').size().reset_index(name='days_worked_last_30')
    active_emps = active_emps.merge(attendance_counts, on='emp_id', how='left').fillna({'days_worked_last_30': 0})
    
    # 4. EWA Features (Last 60 days)
    last_60_days_ewa = ewa_usage[(ewa_usage['date'] > snapshot_dt - timedelta(days=60)) & (ewa_usage['date'] <= snapshot_dt)]
    ewa_stats = last_60_days_ewa.groupby('emp_id').agg({
        'amount': ['count', 'sum', 'mean']
    }).reset_index()
    ewa_stats.columns = ['emp_id', 'ewa_count_60', 'ewa_sum_60', 'ewa_avg_60']
    active_emps = active_emps.merge(ewa_stats, on='emp_id', how='left').fillna(0)
    
    # 5. Target Variable (Resigned in next 3 months?)
    # For training, we need historical snapshots. For inference, target is unknown.
    next_3_months = snapshot_dt + timedelta(days=90)
    resigned_next_3m = resignation[(resignation['resign_date'] > snapshot_dt) & (resignation['resign_date'] <= next_3_months)]['emp_id'].tolist()
    active_emps['resigned_next_3m'] = active_emps['emp_id'].isin(resigned_next_3m).astype(int)
    
    # 6. Categorical Encoding (Simple label encoding for now)
    active_emps['company_cat'] = active_emps['company'].astype('category').cat.codes
    active_emps['dept_cat'] = active_emps['dept'].astype('category').cat.codes
    active_emps['city_cat'] = active_emps['city'].astype('category').cat.codes
    
    # Save processed data
    os.makedirs('ml/data/processed', exist_ok=True)
    active_emps.to_csv(f'ml/data/processed/features_{snapshot_date}.csv', index=False)
    
    print(f"Features processed for {snapshot_date} and saved to ml/data/processed/")
    return active_emps

if __name__ == "__main__":
    # Generate multiple snapshots for training
    process_features('2025-01-01')
    process_features('2025-06-01')
    process_features('2025-12-01')
    process_features('2026-03-01') # Current snapshot for inference
