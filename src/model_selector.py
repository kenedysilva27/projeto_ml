import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import time
from config import ALGORITHMS, RANDOM_STATE

class ModelSelector:
    def __init__(self):
        self.results = {}
        self.best_model = None
    
    def compare_algorithms(self, X, y, sample_size=25000):
        """Compara algoritmos e retorna o melhor"""
        if len(X) > sample_size:
            X, _, y, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=RANDOM_STATE)
        
        models = {
            'LogisticRegression': LogisticRegression(**ALGORITHMS['LogisticRegression']),
            'RandomForest': RandomForestClassifier(**ALGORITHMS['RandomForest']),
            'LightGBM': lgb.LGBMClassifier(**ALGORITHMS['LightGBM'])
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        for name, model in models.items():
            start = time.time()
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            self.results[name] = {
                'model': model,
                'auc': scores.mean(),
                'std': scores.std(),
                'time': time.time() - start
            }
            
            print(f"{name}: {scores.mean():.4f} (±{scores.std():.4f}) - {time.time() - start:.1f}s")
        
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['auc'])
        self.best_model = self.results[best_name]['model']
        
        print(f"Melhor: {best_name}")
        return self.best_model