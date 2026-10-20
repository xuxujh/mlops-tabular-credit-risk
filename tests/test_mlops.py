from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from src.models.trainer import train_and_evaluate
from src.monitoring.drift import detect_feature_drift
from src.serving.app import app


def test_training_pipeline_and_metrics(tmp_path: Path) -> None:
    """Valida o treinamento end-to-end, geração de artefato e qualidade mínima do ROC-AUC."""
    artifact = tmp_path / "model.joblib"
    ref_file = tmp_path / "ref.json"
    metrics = train_and_evaluate(artifact_path=artifact, reference_path=ref_file, n_samples=600)
    assert artifact.exists()
    assert ref_file.exists()
    assert metrics["roc_auc"] > 0.65


def test_ks_drift_detection() -> None:
    """Valida a detecção estatística de Data Drift via teste de Kolmogorov-Smirnov."""
    ref = {"annual_income": [50000.0, 52000.0, 48000.0, 51000.0, 49500.0] * 10}
    cur_no_drift = {"annual_income": [50500.0, 51500.0, 48500.0, 50800.0, 49200.0] * 10}
    cur_drifted = {"annual_income": [180000.0, 195000.0, 210000.0, 175000.0, 205000.0] * 10}

    res_clean = detect_feature_drift(ref, cur_no_drift, alpha=0.05)
    assert res_clean["dataset_drift"] is False

    res_drift = detect_feature_drift(ref, cur_drifted, alpha=0.05)
    assert res_drift["dataset_drift"] is True
    assert res_drift["features"]["annual_income"]["drift_detected"] is True


def test_api_predict_and_drift_endpoints() -> None:
    """Valida os endpoints REST de predição de risco e monitoramento de drift."""
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        applicant = {
            "age": 34,
            "annual_income": 85000.0,
            "debt_to_income": 0.22,
            "credit_score": 740,
            "loan_amount": 12000.0,
            "employment_status": "EMPLOYED",
            "home_ownership": "MORTGAGE",
        }
        pred_resp = client.post("/predict", json=applicant)
        assert pred_resp.status_code == 200
        data = pred_resp.json()
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["risk_tier"] in {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"}
        assert isinstance(data["approved"], bool)

        drift_payload = {"current_data": {"credit_score": [310.0, 320.0, 305.0, 315.0] * 15}}
        drift_resp = client.post("/monitoring/drift", json=drift_payload)
        assert drift_resp.status_code == 200
        assert drift_resp.json()["dataset_drift"] is True