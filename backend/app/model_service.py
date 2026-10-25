"""Model loading + scoring logic.

This is the original `prediction_helper.py`, refactored into a class so it
loads once, supports batches, and can explain each prediction.
The maths is unchanged: scale -> logistic regression -> score = 300 + (1-p) * 600.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_SCORE = 300
SCALE_LENGTH = 600

RATING_BANDS = {
    "Poor": (300, 499),
    "Average": (500, 649),
    "Good": (650, 749),
    "Excellent": (750, 900),
}

# Human-readable names for the model's features
FEATURE_LABELS = {
    "age": "Age",
    "loan_tenure_months": "Loan tenure",
    "number_of_open_accounts": "Open loan accounts",
    "credit_utilization_ratio": "Credit utilization",
    "loan_to_income": "Loan-to-income ratio",
    "delinquency_ratio": "Delinquency ratio",
    "avg_dpd_per_delinquency": "Avg DPD per delinquency",
    "residence_type_Owned": "Residence: Owned",
    "residence_type_Rented": "Residence: Rented",
    "loan_purpose_Education": "Purpose: Education",
    "loan_purpose_Home": "Purpose: Home",
    "loan_purpose_Personal": "Purpose: Personal",
    "loan_type_Unsecured": "Loan type: Unsecured",
}

# Test-set metrics from the notebook (final Logistic Regression, SMOTE-Tomek + Optuna)
NOTEBOOK_METRICS = {
    "auc": 0.983,
    "gini": 0.966,
    "recall_default": 0.93,
    "precision_default": 0.58,
    "f1_default": 0.71,
}

# Maps API field -> model column used for range checks
_RANGE_CHECKS = {
    "age": ("age", "Age"),
    "loan_tenure_months": ("loan_tenure_months", "Loan tenure (months)"),
    "num_open_accounts": ("number_of_open_accounts", "Open loan accounts"),
    "avg_dpd_per_delinquency": ("avg_dpd_per_delinquency", "Avg DPD per delinquency"),
    "loan_to_income": ("loan_to_income", "Loan-to-income ratio"),
}


def get_rating(score: float) -> str:
    if 300 <= score < 500:
        return "Poor"
    if 500 <= score < 650:
        return "Average"
    if 650 <= score < 750:
        return "Good"
    if 750 <= score <= 900:
        return "Excellent"
    return "Undefined"


class CreditRiskModel:
    def __init__(self, model_path: Path):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features: List[str] = list(data["features"])
        self.cols_to_scale: List[str] = list(data["cols_to_scale"])
        self.ranges: Dict[str, Tuple[float, float]] = {
            c: (float(lo), float(hi))
            for c, lo, hi in zip(
                self.cols_to_scale, self.scaler.data_min_, self.scaler.data_max_
            )
        }

    # ---------------------------------------------------------------- features
    @staticmethod
    def _loan_to_income(loan_amount: float, income: float) -> float:
        return loan_amount / income if income > 0 else 0.0

    def _build_frame(self, rows: List[dict]) -> pd.DataFrame:
        """Raw applicant dicts -> scaled model matrix (rows x features)."""
        records = []
        for r in rows:
            records.append(
                {
                    "age": r["age"],
                    "loan_tenure_months": r["loan_tenure_months"],
                    "number_of_open_accounts": r["num_open_accounts"],
                    "credit_utilization_ratio": r["credit_utilization_ratio"],
                    "loan_to_income": self._loan_to_income(r["loan_amount"], r["income"]),
                    "delinquency_ratio": r["delinquency_ratio"],
                    "avg_dpd_per_delinquency": r["avg_dpd_per_delinquency"],
                    "residence_type_Owned": int(r["residence_type"] == "Owned"),
                    "residence_type_Rented": int(r["residence_type"] == "Rented"),
                    "loan_purpose_Education": int(r["loan_purpose"] == "Education"),
                    "loan_purpose_Home": int(r["loan_purpose"] == "Home"),
                    "loan_purpose_Personal": int(r["loan_purpose"] == "Personal"),
                    "loan_type_Unsecured": int(r["loan_type"] == "Unsecured"),
                    # Columns the scaler was fitted on but the final model does not use.
                    # Placeholder values; they are dropped after scaling (same as original app).
                    "number_of_dependants": 1,
                    "years_at_current_address": 1,
                    "zipcode": 1,
                    "sanction_amount": 1,
                    "processing_fee": 1,
                    "gst": 1,
                    "net_disbursement": 1,
                    "principal_outstanding": 1,
                    "bank_balance_at_application": 1,
                    "number_of_closed_accounts": 1,
                    "enquiry_count": 1,
                }
            )
        df = pd.DataFrame(records)
        df[self.cols_to_scale] = self.scaler.transform(df[self.cols_to_scale])
        return df[self.features]

    # ----------------------------------------------------------------- scoring
    def _score(self, X: pd.DataFrame):
        logits = X.values @ self.model.coef_.T + self.model.intercept_
        prob = (1 / (1 + np.exp(-logits))).flatten()
        score = BASE_SCORE + (1 - prob) * SCALE_LENGTH
        return logits.flatten(), prob, score

    def predict_many(self, rows: List[dict]) -> List[dict]:
        X = self._build_frame(rows)
        _, prob, score = self._score(X)
        out = []
        for r, p, s in zip(rows, prob, score):
            out.append(
                {
                    "default_probability": float(p),
                    "credit_score": int(s),
                    "rating": get_rating(s),
                    "loan_to_income": round(self._loan_to_income(r["loan_amount"], r["income"]), 2),
                }
            )
        return out

    def predict_one(self, row: dict) -> dict:
        X = self._build_frame([row])
        _, prob, score = self._score(X)
        lti = self._loan_to_income(row["loan_amount"], row["income"])

        # coef * scaled value = each feature's push on the log-odds (intercept excluded)
        contrib = X.values[0] * self.model.coef_[0]
        contributions = [
            {"feature": f, "label": FEATURE_LABELS.get(f, f), "value": float(c)}
            for f, c in zip(self.features, contrib)
        ]
        contributions.sort(key=lambda d: abs(d["value"]), reverse=True)

        return {
            "default_probability": float(prob[0]),
            "credit_score": int(score[0]),
            "rating": get_rating(score[0]),
            "loan_to_income": round(lti, 2),
            "warnings": self.range_warnings(row, lti),
            "contributions": contributions,
        }

    # ---------------------------------------------------------------- guard rails
    def range_warnings(self, row: dict, lti: float) -> List[str]:
        values = dict(row)
        values["loan_to_income"] = lti
        msgs = []
        for key, (col, label) in _RANGE_CHECKS.items():
            lo, hi = self.ranges[col]
            v = values[key]
            if v < lo or v > hi:
                msgs.append(
                    f"{label} = {v:g} is outside the training range ({lo:g}–{hi:g}); "
                    "the prediction is an extrapolation and may be unreliable."
                )
        return msgs

    def info(self) -> dict:
        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.features),
            "features": self.features,
            "training_ranges": {
                c: {"min": lo, "max": hi} for c, (lo, hi) in self.ranges.items()
            },
            "score_range": [BASE_SCORE, BASE_SCORE + SCALE_LENGTH],
            "rating_bands": {k: list(v) for k, v in RATING_BANDS.items()},
            "metrics": NOTEBOOK_METRICS,
        }
