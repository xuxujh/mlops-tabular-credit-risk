# MLOps Tabular Credit Risk & Drift Monitoring

Pipeline **MLOps end-to-end** para modelagem preditiva de risco de crédito, engenharia de features automatizada, detecção estatística de *Data Drift* e API de inferência em tempo real utilizando **Scikit-learn**, **SciPy**, **FastAPI** e **Pydantic v2**.

---

## Arquitetura do Pipeline MLOps

```
mermaid
graph LR
    A[Dados Tabulares / Payload] -->|Validação Pydantic v2| B(ColumnTransformer: Imputer + Scaler + OneHot)
    B -->|Features Processadas| C[GradientBoostingClassifier]
    C -->|Probabilidade & Tier| D[FastAPI Endpoint /predict]
    A -->|Distribuição em Produção| E[Monitoramento KS-Test SciPy]
    E -->|p-value < 0.05| F[Alerta de Data Drift /monitoring/drift]
```

---

## Stack Tecnológica

* **Machine Learning & Feature Engineering:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `GradientBoostingClassifier`).
* **Monitoramento de Data Drift:** `scipy.stats.ks_2samp` (Teste não-paramétrico de Kolmogorov-Smirnov para comparação entre distribuições de treino e produção).
* **Contratos de Dados & Serialização:** `Pydantic v2`, `Joblib`.
* **Serving & Produção:** `FastAPI`, `Uvicorn`, `Docker` (multi-step com pré-treinamento de artefato em build) e **GitHub Actions**.

---

## Como Executar Localmente

```
powershell
# 1. Treinar o modelo e gerar artefatos de referência
python -m src.models.trainer

# 2. Executar linter e suíte de testes com cobertura
python -m ruff check src/ tests/
python -m pytest --cov=src tests/

# 3. Iniciar a API de Inferência e Monitoramento
python -m src.serving.app
```