from fastapi import FastAPI, HTTPException
from typing import List, Dict
import pandas as pd
from src.predictor import CustomerPredictor

predictor = CustomerPredictor()
app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"model_loaded": predictor.is_loaded}

@app.post("/predict")
def predict(data: Dict[str, float]):
    """Predição para um cliente"""
    if not predictor.is_loaded:
        predictor.load_model()
    
    result = predictor.predict_single(data)
    return {
        "prediction": bool(result['prediction']),
        "probability": float(result['probability']),
        "confidence": float(result['confidence'])
    }

@app.post("/predict_batch")  
def predict_batch(data: List[Dict[str, float]]):
    """Predições para múltiplos clientes"""
    if not predictor.is_loaded:
        predictor.load_model()
    
    results = predictor.predict_batch(pd.DataFrame(data))
    return [
        {
            "prediction": bool(row['prediction']),
            "probability": float(row['probability']),
            "confidence": float(row['confidence'])
        }
        for _, row in results.iterrows()
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)