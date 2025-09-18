import pandas as pd
import joblib
import os
from src.feature_engineer import FeatureEngineer
from src.model_selector import ModelSelector
from config import TRAIN_DATA_PATH, MODEL_PATH, FEATURE_ENGINEER_PATH

def train_model():
    """Pipeline de treinamento do modelo"""
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Erro: Arquivo {TRAIN_DATA_PATH} não encontrado!")
        return False
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    feature_cols = [col for col in train_df.columns if col.startswith('var_')]
    X = train_df[feature_cols].copy()
    y = train_df['target'].copy()
    
    feature_eng = FeatureEngineer()
    X_engineered = feature_eng.create_features(X)
    
    model_selector = ModelSelector()
    model_selector.compare_algorithms(X_engineered, y)
    best_model = model_selector.best_model
    
    best_model.fit(X_engineered, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_eng, FEATURE_ENGINEER_PATH)
    
    return True

if __name__ == "__main__":
    train_model()
