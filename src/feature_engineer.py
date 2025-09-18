import pandas as pd

class FeatureEngineer:
    """Cria features a partir das variáveis originais"""
    
    def __init__(self):
        self.created_features = []
    
    def create_features(self, df):
        """Cria features estatísticas"""
        var_cols = [col for col in df.columns if col.startswith('var_')]
        
        df['mean_features'] = df[var_cols].mean(axis=1)
        df['std_features'] = df[var_cols].std(axis=1)
        df['min_features'] = df[var_cols].min(axis=1)
        df['max_features'] = df[var_cols].max(axis=1)
        df['count_positive'] = (df[var_cols] > 0).sum(axis=1)
        df['count_negative'] = (df[var_cols] < 0).sum(axis=1)
        total_features = len(var_cols)
        df['prop_positive'] = df['count_positive'] / total_features
        df['prop_negative'] = df['count_negative'] / total_features
        
        self.created_features = [
            'mean_features', 'std_features', 'min_features', 'max_features',
            'count_positive', 'count_negative', 'prop_positive', 'prop_negative'
        ]
        
        return df
    
    def get_created_features(self):
        return self.created_features
