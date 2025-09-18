import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineer import FeatureEngineer
from src.model_selector import ModelSelector
from src.predictor import CustomerPredictor

class TestIntegration:
    """Testes de integração do pipeline completo"""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({f'var_{i}': np.random.randn(n) for i in range(200)})
        score = df[['var_0', 'var_1', 'var_2']].sum(axis=1) + np.random.randn(n) * 0.1
        df['target'] = (score > score.median()).astype(int)
        return df

    def test_feature_engineering(self, sample_data):
        """Verifica se as features são criadas corretamente"""
        X = sample_data.filter(like='var_')
        fe = FeatureEngineer()
        X_new = fe.create_features(X.copy())

        created = fe.get_created_features()
        assert all(col in X_new.columns for col in created)
        assert not X_new[created].isnull().any().any()

    def test_model_selection(self, sample_data):
        """Verifica se o modelo é selecionado corretamente"""
        X = sample_data.filter(like='var_')
        y = sample_data['target']
        fe = FeatureEngineer()
        X_new = fe.create_features(X.copy())

        ms = ModelSelector()
        ms.compare_algorithms(X_new, y, sample_size=500)

        assert ms.best_model is not None

    def test_training_and_prediction(self, sample_data):
        """Testa o pipeline completo de treino e predição"""
        X = sample_data.filter(like='var_')
        y = sample_data['target']
        fe = FeatureEngineer()
        X_new = fe.create_features(X.copy())

        ms = ModelSelector()
        ms.compare_algorithms(X_new, y, sample_size=500)
        model = ms.best_model
        model.fit(X_new, y)

        predictor = CustomerPredictor()
        predictor.model = model
        predictor.feature_engineer = fe
        predictor.is_loaded = True

        result = predictor.predict_single(X.iloc[0])
        assert set(result) == {'prediction', 'probability', 'confidence'}

        batch = predictor.predict_batch(X.head(10))
        assert len(batch) == 10
        assert 'prediction' in batch.columns

    def test_prediction_from_file(self, sample_data):
        """Testa predição a partir de arquivo CSV"""
        with tempfile.TemporaryDirectory() as tmp:
            df = sample_data.head(50).copy()
            df['ID_code'] = [f'test_{i}' for i in range(50)]
            input_path = os.path.join(tmp, 'input.csv')
            output_path = os.path.join(tmp, 'output.csv')
            df.drop('target', axis=1).to_csv(input_path, index=False)

            class MockModel:
                def predict_proba(self, X):
                    return np.random.rand(len(X), 2)

            predictor = CustomerPredictor()
            predictor.model = MockModel()
            predictor.feature_engineer = FeatureEngineer()
            predictor.is_loaded = True

            results = predictor.predict_from_file(input_path, output_path)
            assert len(results) == 50
            saved = pd.read_csv(output_path)
            assert len(saved) == 50
            assert {'ID_code', 'prediction'}.issubset(saved.columns)
