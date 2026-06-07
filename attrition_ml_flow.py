import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import random

# ==========================================
# 1. DATA SIMULATION
# ==========================================
class DataSimulator:
    def __init__(self, n_employees=1000):
        self.n_employees = n_employees
        self.companies = ['TechCorp', 'RetailCo', 'FactoryX', 'SalesForce']
        self.industries = ['Technology', 'Retail', 'Manufacturing', 'Sales']
        self.departments = ['HR', 'IT', 'Sales', 'Production', 'Finance']
        self.roles = ['Staff', 'Lead', 'Manager', 'Worker']
        
    def generate_employees(self):
        data = []
        start_date_range = pd.date_range(start='2023-01-01', end='2024-06-01')
        
        for i in range(self.n_employees):
            emp_id = f'EMP_{i:04d}'
            gender = random.choice(['Male', 'Female'])
            dob = datetime(1980, 1, 1) + timedelta(days=random.randint(0, 365*20))
            start_date = random.choice(start_date_range)
            company = random.choice(self.companies)
            industry = self.industries[self.companies.index(company)]
            dept = random.choice(self.departments)
            role = random.choice(self.roles)
            
            # Probability of attrition (some intrinsic risk)
            base_risk = random.uniform(0.01, 0.1)
            
            # Resign date (if they resign)
            resign_date = None
            if random.random() < 0.15: # 15% overall attrition rate in simulation
                resign_days = random.randint(30, 400)
                resign_date = start_date + timedelta(days=resign_days)
                if resign_date > datetime(2025, 3, 1): # Cap at current date
                    resign_date = None
            
            data.append({
                'employee_id': emp_id,
                'gender': gender,
                'dob': dob,
                'start_date': start_date,
                'resign_date': resign_date,
                'company': company,
                'industry': industry,
                'department': dept,
                'role': role,
                'base_risk': base_risk,
                'salary': random.randint(10, 50) * 1000000 # 10M - 50M VND
            })
        return pd.DataFrame(data)

    def generate_attendance(self, employees, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date)
        attendance_data = []
        
        for _, emp in employees.iterrows():
            emp_start = emp['start_date']
            emp_end = emp['resign_date'] if emp['resign_date'] else end_date
            
            active_dates = [d for d in dates if emp_start <= d <= emp_end]
            
            for d in active_dates:
                # Sunday off
                if d.weekday() == 6: continue
                
                # Random absenteeism
                status = 'work'
                if random.random() < 0.05: # 5% chance of leave
                    status = random.choice(['unpaid_leave', 'sick', 'annual_leave'])
                
                # Higher absenteeism before resignation
                if emp['resign_date'] and (emp['resign_date'] - d).days < 30:
                    if random.random() < 0.2: # 20% chance of leave in last month
                        status = 'unpaid_leave'
                
                attendance_data.append({
                    'employee_id': emp['employee_id'],
                    'date': d,
                    'status': status,
                    'ot_hours': random.randint(0, 4) if random.random() > 0.7 else 0
                })
        return pd.DataFrame(attendance_data)

    def generate_app_activity(self, employees, start_date, end_date):
        activity = []
        dates = pd.date_range(start=start_date, end=end_date)
        
        for _, emp in employees.iterrows():
            emp_start = emp['start_date']
            emp_end = emp['resign_date'] if emp['resign_date'] else end_date
            
            # App registration
            reg_date = emp_start + timedelta(days=random.randint(0, 30))
            if reg_date > end_date: continue
            
            active_dates = [d for d in dates if reg_date <= d <= emp_end]
            
            for d in active_dates:
                # Login frequency
                if random.random() < 0.3: # Log in 30% of days
                    activity.append({
                        'employee_id': emp['employee_id'],
                        'date': d,
                        'type': 'access',
                        'amount': 0
                    })
                
                # Transactions (EWA)
                if random.random() < 0.1: # Transact 10% of days
                    amount = random.randint(5, 20) * 100000 # 500k - 2M
                    activity.append({
                        'employee_id': emp['employee_id'],
                        'date': d,
                        'type': 'transaction',
                        'amount': amount
                    })
                    
            # Behavioral change: high risk users might stop using app or increase withdrawals
            # (Simplified for simulation)
            
        return pd.DataFrame(activity)

# ==========================================
# 2. ML PIPELINE
# ==========================================
class AttritionPipeline:
    def __init__(self, employees, attendance, app_activity):
        self.employees = employees
        self.attendance = attendance
        self.app_activity = app_activity
        self.model = None
        self.features = [
            'age', 'tenure_days', 'work_ratio', 'unpaid_leave_ratio', 
            'ot_avg', 'app_access_freq', 'avg_tx_amount', 'salary'
        ]

    def create_snapshot(self, snapshot_date):
        # 1. Filter active employees at snapshot
        active = self.employees[
            (self.employees['start_date'] <= snapshot_date) & 
            ((self.employees['resign_date'].isnull()) | (self.employees['resign_date'] > snapshot_date))
        ].copy()
        
        if active.empty: return pd.DataFrame()

        # 2. Basic features
        active['age'] = (snapshot_date - active['dob']).dt.days // 365
        active['tenure_days'] = (snapshot_date - active['start_date']).dt.days
        
        # 3. Attendance features (last 30 days)
        window_start = snapshot_date - timedelta(days=30)
        att_window = self.attendance[
            (self.attendance['date'] >= window_start) & (self.attendance['date'] < snapshot_date)
        ]
        
        att_stats = att_window.groupby('employee_id').agg(
            total_days=('status', 'count'),
            work_days=('status', lambda x: (x == 'work').sum()),
            unpaid_days=('status', lambda x: (x == 'unpaid_leave').sum()),
            ot_total=('ot_hours', 'sum')
        ).reset_index()
        
        active = active.merge(att_stats, on='employee_id', how='left').fillna(0)
        active['work_ratio'] = active['work_days'] / active['total_days'].replace(0, 1)
        active['unpaid_leave_ratio'] = active['unpaid_days'] / active['total_days'].replace(0, 1)
        active['ot_avg'] = active['ot_total'] / active['total_days'].replace(0, 1)
        
        # 4. App features (last 30 days)
        app_window = self.app_activity[
            (self.app_activity['date'] >= window_start) & (self.app_activity['date'] < snapshot_date)
        ]
        
        app_stats = app_window.groupby('employee_id').agg(
            access_count=('type', lambda x: (x == 'access').sum()),
            tx_count=('type', lambda x: (x == 'transaction').sum()),
            tx_sum=('amount', 'sum')
        ).reset_index()
        
        active = active.merge(app_stats, on='employee_id', how='left').fillna(0)
        active['app_access_freq'] = active['access_count'] / 30
        active['avg_tx_amount'] = active['tx_sum'] / active['tx_count'].replace(0, 1)
        
        # 5. Labeling (Resign in next 90 days?)
        def get_label(row):
            res_date = row['resign_date']
            if pd.isnull(res_date): return 0
            if snapshot_date < res_date <= (snapshot_date + timedelta(days=90)):
                return 1
            return 0
            
        active['label'] = active.apply(get_label, axis=1)
        active['snapshot_date'] = snapshot_date
        
        return active

    def build_training_data(self, start_date, end_date):
        snapshots = []
        for date in pd.date_range(start=start_date, end=end_date, freq='MS'):
            snap = self.create_snapshot(date)
            if not snap.empty:
                snapshots.append(snap)
        return pd.concat(snapshots)

    def train(self, train_df):
        X = train_df[self.features]
        y = train_df['label']
        
        # Simple LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=5, # Handle imbalance
            importance_type='gain'
        )
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict(self, current_date):
        snap = self.create_snapshot(current_date)
        if snap.empty: return pd.DataFrame()
        
        X = snap[self.features]
        probs = self.model.predict_proba(X)[:, 1]
        
        snap['attrition_probability'] = probs
        # Risk Tiers
        snap['risk_tier'] = pd.qcut(probs, q=[0, 0.7, 0.9, 1.0], labels=['Low', 'Medium', 'High'], duplicates='drop')
        
        return snap[['employee_id', 'snapshot_date', 'attrition_probability', 'risk_tier']]

# ==========================================
# 3. EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # A. INITIAL SETUP (THIẾT LẬP BAN ĐẦU)
    print("--- Phase 1: Simulating Data ---")
    sim = DataSimulator(n_employees=2000)
    employees = sim.generate_employees()
    attendance = sim.generate_attendance(employees, '2023-01-01', '2025-03-01')
    app_activity = sim.generate_app_activity(employees, '2023-01-01', '2025-03-01')
    
    pipeline = AttritionPipeline(employees, attendance, app_activity)
    
    # B. TRAINING (HUẤN LUYỆN LẦN ĐẦU)
    print("\n--- Phase 2: Building Training Dataset (Jan 2024 - Dec 2024) ---")
    train_data = pipeline.build_training_data('2024-01-01', '2024-12-01')
    pipeline.train(train_data)
    
    # C. MONTHLY RE-RUN (CHẠY ĐỊNH KỲ HÀNG THÁNG)
    print("\n--- Phase 3: Monthly Re-run Simulation ---")
    
    # Run for Jan 2025
    jan_results = pipeline.predict(datetime(2025, 1, 1))
    print(f"Jan 2025 Results: {len(jan_results)} employees scored.")
    print(jan_results['risk_tier'].value_counts())
    
    # Run for Feb 2025
    feb_results = pipeline.predict(datetime(2025, 2, 1))
    print(f"\nFeb 2025 Results: {len(feb_results)} employees scored.")
    print(feb_results['risk_tier'].value_counts())
    
    # D. OUTPUT (LƯU KẾT QUẢ)
    # jan_results.to_csv('attrition_results_2025_01.csv', index=False)
    print("\nFlow completed. Results ready for Dashboard.")
