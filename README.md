# Predição de Transações
Projeto de Machine Learning para prever se um cliente realizará uma transação, com base em 200 variáveis numéricas. A solução inclui pipeline de treinamento, geração de features, seleção de modelo, API de predição e testes automatizados.

## Instalação

pip install -r requirements.txt

## Treinamento

python train.py

O modelo treinado será salvo na pasta models/

## Executando a API

uvicorn api:app --reload ou python api.py

Acesso a documentação em: http://localhost:8000/docs

## Endpoints

GET /health – Verifica se o modelo está carregado

POST /predict – Predição para um cliente

POST /predict_batch – Predições em lote

Exemplo de uso:

import requests

data = {f"var_{i}": 0.1 for i in range(200)}
res = requests.post("http://localhost:8000/predict", json=data)
print(res.json())

## Features criadas

A partir das 200 variáveis originais, são geradas 8 novas features estatísticas:

Média, desvio padrão, mínimo, máximo

Contagem e proporção de valores positivos e negativos

## Algoritmos testados

Logistic Regression

Random Forest

LightGBM

O melhor modelo é selecionado automaticamente com base no AUC-ROC.

## Testes

Execute todos os testes com:

pytest -v

Testes incluídos:

Unitários (CustomerPredictor)

Integração (pipeline completo)

API (validação dos endpoints)

## Docker

O projeto pode ser containerizado com Docker.

Dockerfile

FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

.dockerignore

__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
.git/
*.log
*.csv
*.ipynb
data/
models/
tests/

## Como usar

docker build -t imagem-ml .
docker run -p 8000:8000 imagem-ml

## Estrutura do Projeto

projeto_ml/
├── api.py
├── train.py
├── main.py
├── models/
├── data/
├── src/
│   ├── feature_engineer.py
│   ├── model_selector.py
│   ├── predictor.py
├── tests/
│   ├── test_unitario.py
│   ├── test_integration.py
│   ├── test_api.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md

