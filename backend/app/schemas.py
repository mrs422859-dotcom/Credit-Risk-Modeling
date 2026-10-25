"""Pydantic request / response models for the Credit Risk API."""
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ResidenceType(str, Enum):
    owned = "Owned"
    rented = "Rented"
    mortgage = "Mortgage"


class LoanPurpose(str, Enum):
    education = "Education"
    home = "Home"
    auto = "Auto"
    personal = "Personal"


class LoanType(str, Enum):
    unsecured = "Unsecured"
    secured = "Secured"


class Rating(str, Enum):
    poor = "Poor"
    average = "Average"
    good = "Good"
    excellent = "Excellent"


class CreditApplication(BaseModel):
    """Raw applicant inputs (same fields as the original Streamlit form)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 28,
                "income": 1200000,
                "loan_amount": 2560000,
                "loan_tenure_months": 36,
                "avg_dpd_per_delinquency": 20,
                "delinquency_ratio": 30,
                "credit_utilization_ratio": 30,
                "num_open_accounts": 2,
                "residence_type": "Owned",
                "loan_purpose": "Home",
                "loan_type": "Secured",
            }
        }
    )

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    income: float = Field(..., gt=0, description="Annual income (INR)")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount (INR)")
    loan_tenure_months: int = Field(..., ge=1, le=360, description="Loan tenure in months")
    avg_dpd_per_delinquency: float = Field(
        ..., ge=0, description="Average days-past-due per delinquent month"
    )
    delinquency_ratio: float = Field(
        ..., ge=0, le=100, description="Delinquent months / total loan months, in %"
    )
    credit_utilization_ratio: float = Field(
        ..., ge=0, le=100, description="Credit utilization in %"
    )
    num_open_accounts: int = Field(..., ge=1, le=10, description="Open loan accounts")
    residence_type: ResidenceType
    loan_purpose: LoanPurpose
    loan_type: LoanType


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float = Field(..., description="Contribution to the log-odds of default")


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    credit_score: int = Field(..., ge=300, le=900)
    rating: Rating
    loan_to_income: float
    warnings: List[str] = Field(
        default_factory=list,
        description="Inputs that fall outside the range the model was trained on",
    )
    contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Per-feature contribution to log-odds (positive = raises default risk)",
    )


class BatchRequest(BaseModel):
    applications: List[CreditApplication] = Field(..., min_length=1, max_length=1000)


class BatchItem(BaseModel):
    index: int
    default_probability: float
    credit_score: int
    rating: Rating
    loan_to_income: float


class BatchResponse(BaseModel):
    count: int
    results: List[BatchItem]


class FeatureRange(BaseModel):
    min: float
    max: float


class ModelInfo(BaseModel):
    model_type: str
    n_features: int
    features: List[str]
    training_ranges: Dict[str, FeatureRange]
    score_range: List[int]
    rating_bands: Dict[str, List[int]]
    metrics: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
