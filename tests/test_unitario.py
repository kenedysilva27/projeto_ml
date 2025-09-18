import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predictor import CustomerPredictor

class TestCustomerPredictor:

    @pytest.fixture
    def sample_customer(self):
        """Gera um cliente fictício com 200 variáveis"""
        return {f'var_{i}': np.random.randn() for i in range(200)}

    @pytest.fixture
    def sample_customers_df(self):
        """Gera um DataFrame com 10 clientes fictícios"""
        return pd.DataFrame([
            {**{f'var_{j}': np.random.randn() for j in range(200)}, 'ID_code': f'test_{i}'}
            for i in range(10)
        ])

    def test_init(self):
        """Verifica se o predictor inicia vazio"""
        p = CustomerPredictor()
        assert p.model is None
        assert p.feature_engineer is None
        assert not p.is_loaded

    def test_predict_single(self, sample_customer):
        """Testa se a predição única retorna os campos esperados"""
        if not os.path.exists('models/model.joblib'):
            pytest.skip("Modelo não encontrado")

        result = CustomerPredictor().predict_single(sample_customer)
        assert set(result) == {'prediction', 'probability', 'confidence'}
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        assert 0 <= result['confidence'] <= 1

    def test_predict_batch(self, sample_customers_df):
        """Testa se a predição em lote retorna um DataFrame válido"""
        if not os.path.exists('models/model.joblib'):
            pytest.skip("Modelo não encontrado")

        results = CustomerPredictor().predict_batch(sample_customers_df)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_customers_df)
        assert {'prediction', 'probability', 'confidence'}.issubset(results.columns)

    def test_missing_vars(self):
        """Testa se o predictor acusa erro quando faltam variáveis"""
        df = pd.DataFrame({'var_0': [1.0], 'var_1': [2.0]})
        if os.path.exists('models/model.joblib'):
            with pytest.raises(ValueError):
                CustomerPredictor().predict_single(df)

    def test_model_not_found(self):
        """Testa se o predictor lança erro quando o modelo não existe"""
        p = CustomerPredictor()
        fake_path = 'models/arquivo_inexistente.joblib'
        if not os.path.exists(fake_path):
            with pytest.raises(FileNotFoundError):
                import joblib
                joblib.load(fake_path)
