import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os
import json

class AttritionModel:
    def __init__(self, model_type='RandomForest'):
        self.model_type = model_type
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'GradientBoosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'LogisticRegression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_prob))
        }
        return metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

def compare_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare their performance.
    """
    model_types = ['RandomForest', 'GradientBoosting', 'LogisticRegression']
    results = {}
    
    for m_type in model_types:
        print(f"Training {m_type}...")
        model = AttritionModel(m_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[m_type] = metrics
        
    # Save comparison results
    os.makedirs('ml/artifacts', exist_ok=True)
    with open('ml/artifacts/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Pick the best model based on F1 score
    best_model_type = max(results, key=lambda x: results[x]['f1'])
    print(f"Best model: {best_model_type}")
    
    # Train and save the best model
    best_model = AttritionModel(best_model_type)
    best_model.train(X_train, y_train)
    best_model.save('ml/models/best_model.pkl')
    
    return best_model_type, results

if __name__ == "__main__":
    # Example usage (will be called from train_eval.py)
    pass
