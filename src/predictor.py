import sys
import os

import pandas as pd
import numpy as np
import joblib
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_PATH, FEATURE_ENGINEER_PATH, PREDICTION_THRESHOLD, BATCH_SIZE

class CustomerPredictor:
    """Faz predições de transações de clientes"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.threshold = PREDICTION_THRESHOLD
        self.is_loaded = False
    
    def load_model(self):
        """Carrega o modelo treinado"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
        
        if not os.path.exists(FEATURE_ENGINEER_PATH):
            raise FileNotFoundError(f"Feature engineer não encontrado: {FEATURE_ENGINEER_PATH}")
        
        self.model = joblib.load(MODEL_PATH)
        self.feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
        self.is_loaded = True
    
    def _prepare_data(self, data):
        """Prepara os dados aplicando feature engineering"""
        if not self.is_loaded:
            self.load_model()
        
        expected_vars = [f'var_{i}' for i in range(200)]
        missing_vars = [var for var in expected_vars if var not in data.columns]
        if missing_vars:
            raise ValueError(f"Variáveis faltando: {missing_vars[:5]}...")
        
        data_processed = data[expected_vars].copy()
        return self.feature_engineer.create_features(data_processed)
    
    def predict_single(self, customer_data):
        """Predição para um único cliente"""
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.Series):
            df = customer_data.to_frame().T
        else:
            df = customer_data.copy()
        
        features = self._prepare_data(df)
        
        probability = self.model.predict_proba(features)[0, 1]
        prediction = int(probability > self.threshold)
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2)
        }
    
    def predict_batch(self, customers_data):
        """Predições em lote para múltiplos clientes"""
        
        features = self._prepare_data(customers_data)
        
        predictions = []
        probabilities = []
        
        for i in range(0, len(features), BATCH_SIZE):
            batch = features.iloc[i:i+BATCH_SIZE]
            batch_probs = self.model.predict_proba(batch)[:, 1]
            batch_preds = (batch_probs > self.threshold).astype(int)
            
            predictions.extend(batch_preds)
            probabilities.extend(batch_probs)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'confidence': [abs(p - 0.5) * 2 for p in probabilities]
        })
        
        if 'ID_code' in customers_data.columns:
            results.insert(0, 'ID_code', customers_data['ID_code'].values)
        
        return results
    
    def predict_from_file(self, input_file, output_file=None):
        """Predições a partir de arquivo CSV"""
        data = pd.read_csv(input_file)
        results = self.predict_batch(data)
        
        if output_file:
            results.to_csv(output_file, index=False)
        
        return results
