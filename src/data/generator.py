from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CreditApplicantRecord(BaseModel):
    """Contrato de dados individual para entrada de inferência e validação."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0.0)
    debt_to_income: float = Field(..., ge=0.0, le=1.5)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0.0)
    employment_status: str = Field(..., pattern="^(EMPLOYED|SELF_EMPLOYED|UNEMPLOYED)$")
    home_ownership: str = Field(..., pattern="^(RENT|MORTGAGE|OWN)$")


def generate_credit_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Gera dataset sintético reproduzível de risco de crédito."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n_samples)
    annual_income = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n_samples), 2)
    debt_to_income = np.round(rng.uniform(0.05, 0.85, size=n_samples), 4)
    credit_score = rng.integers(350, 850, size=n_samples)
    loan_amount = np.round(rng.uniform(2000.0, 60000.0, size=n_samples), 2)
    employment_status = rng.choice(
        ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED"],
        size=n_samples,
        p=[0.70, 0.20, 0.10],
    )
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"],
        size=n_samples,
        p=[0.45, 0.40, 0.15],
    )

    # Log-odds sintético para probabilidade de default
    logit = (
        -2.2
        + 2.8 * debt_to_income
        - 0.005 * (credit_score - 600)
        + 0.00003 * loan_amount
        - 0.00001 * annual_income
        + np.where(employment_status == "UNEMPLOYED", 1.1, 0.0)
        + np.where(home_ownership == "RENT", 0.35, 0.0)
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    default_target = rng.binomial(1, prob_default, size=n_samples)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "debt_to_income": debt_to_income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_status": employment_status,
            "home_ownership": home_ownership,
            "default": default_target,
        }
    )