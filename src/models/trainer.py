from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.generator import generate_credit_dataset
from src.features.pipeline import NUMERIC_FEATURES, build_preprocessor


def build_model_pipeline(random_state: int = 42) -> Pipeline:
    """Combina o pré-processador e o classificador Gradient Boosting em um Pipeline único."""
    preprocessor = build_preprocessor()
    classifier = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(
    artifact_path: str | Path = "models/credit_risk_pipeline.joblib",
    reference_path: str | Path = "models/reference_data.json",
    n_samples: int = 1500,
    random_state: int = 42,
) -> dict[str, Any]:
    """Executa o treinamento, calcula métricas de validação e persiste os artefatos."""
    df = generate_credit_dataset(n_samples=n_samples, random_state=random_state)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = build_model_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_file)

    # Salva amostra de referência das features numéricas para cálculo de Data Drift (KS Test)
    ref_file = Path(reference_path)
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_samples = {
        col: X_train[col].iloc[:300].tolist() for col in NUMERIC_FEATURES
    }
    ref_file.write_text(json.dumps(ref_samples), encoding="utf-8")

    return metrics


def load_or_train_pipeline(
    artifact_path: str | Path = "models/credit_risk_pipeline.joblib",
) -> Pipeline:
    """Carrega o pipeline serializado ou treina um novo caso o artefato não exista."""
    path = Path(artifact_path)
    if not path.exists():
        train_and_evaluate(artifact_path=path)
    return joblib.load(path)


if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"Treinamento concluído com sucesso: {results}")