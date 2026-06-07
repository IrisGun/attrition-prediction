import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def generate_raw_data(num_employees=2000, start_date='2024-01-01', end_date='2026-03-01'):
    """
    Generate synthetic employee data, attendance, and EWA usage.
    """
    np.random.seed(42)
    
    # 1. Employee Master
    emp_ids = [f'EMP_{i:04d}' for i in range(num_employees)]
    companies = ['TechCorp', 'RetailCo', 'FactoryX', 'SalesForce']
    departments = ['Production', 'Sales', 'HR', 'IT', 'Finance', 'Logistics']
    
    employee_master = pd.DataFrame({
        'emp_id': emp_ids,
        'company': np.random.choice(companies, num_employees),
        'dept': np.random.choice(departments, num_employees),
        'join_date': [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=np.random.randint(0, 730)) for _ in range(num_employees)],
        'birth_year': np.random.randint(1980, 2005, num_employees),
        'gender': np.random.choice(['M', 'F'], num_employees),
        'city': np.random.choice(['Hanoi', 'HCM', 'Binh Duong', 'Dong Nai'], num_employees)
    })
    
    # 2. Daily Attendance (Sampled)
    attendance_records = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_dt:
        # Only work days (Mon-Sat)
        if current_date.weekday() < 6:
            # Randomly pick 90-95% of employees present
            present_count = int(num_employees * np.random.uniform(0.9, 0.95))
            present_emps = np.random.choice(emp_ids, present_count, replace=False)
            
            for emp_id in present_emps:
                attendance_records.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'emp_id': emp_id,
                    'check_in': '08:00',
                    'check_out': '17:00',
                    'overtime_hours': np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
                })
        current_date += timedelta(days=1)
        
    attendance_df = pd.DataFrame(attendance_records)
    
    # 3. EWA Usage (Earned Wage Access)
    ewa_records = []
    for emp_id in emp_ids:
        # Some employees use EWA frequently, some never
        usage_freq = np.random.choice([0, 1, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        if usage_freq > 0:
            for _ in range(np.random.randint(1, 50)):
                ewa_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=np.random.randint(0, 800))
                if ewa_date <= end_dt:
                    ewa_records.append({
                        'date': ewa_date.strftime('%Y-%m-%d'),
                        'emp_id': emp_id,
                        'amount': np.random.randint(500000, 2000000)
                    })
    
    ewa_df = pd.DataFrame(ewa_records)
    
    # 4. Resignation (Ground Truth for training)
    # Higher risk for: short tenure, high EWA usage, decreasing attendance
    resignation_records = []
    for emp_id in emp_ids:
        # Simulate some resignations
        if np.random.rand() < 0.15: # 15% attrition rate
            resign_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=np.random.randint(100, 800))
            if resign_date <= end_dt:
                resignation_records.append({
                    'emp_id': emp_id,
                    'resign_date': resign_date.strftime('%Y-%m-%d'),
                    'reason': np.random.choice(['Personal', 'Better Offer', 'Health', 'Relocation'])
                })
    
    resignation_df = pd.DataFrame(resignation_records)
    
    # Save to raw
    os.makedirs('ml/data/raw', exist_ok=True)
    employee_master.to_csv('ml/data/raw/employee_master.csv', index=False)
    attendance_df.to_csv('ml/data/raw/attendance.csv', index=False)
    ewa_df.to_csv('ml/data/raw/ewa_usage.csv', index=False)
    resignation_df.to_csv('ml/data/raw/resignation.csv', index=False)
    
    print("Raw data generated successfully in ml/data/raw/")

if __name__ == "__main__":
    generate_raw_data()
