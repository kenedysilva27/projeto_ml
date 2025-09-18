# Configurações
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.5
BATCH_SIZE = 10000

# Caminhos dos arquivos
MODEL_PATH = 'models/model.joblib'
FEATURE_ENGINEER_PATH = 'models/feature_engineer.joblib'
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'

# Configurações dos algoritmos e seus hiperparâmetros
SCALE_POS_WEIGHT = 7.29 

ALGORITHMS = {
    'LogisticRegression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced' 
    },
    'RandomForest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'class_weight': 'balanced'  
    },
    'LightGBM': {
        'n_estimators': 200,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'scale_pos_weight': SCALE_POS_WEIGHT 
    }
}
