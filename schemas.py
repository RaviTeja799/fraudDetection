from pydantic import BaseModel, Field
from typing import Literal


class TransactionInput(BaseModel):
    kyc_verified: Literal["Yes", "No"]
    account_age_days: int = Field(..., ge=0, description="Age of account in days")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in INR")
    channel: Literal["ATM", "Mobile", "POS", "Web"]
    timestamp: str = Field(..., description="ISO format, e.g. 2025-08-12T02:10:24")

    class Config:
        json_schema_extra = {
            "example": {
                "kyc_verified": "No",
                "account_age_days": 120,
                "transaction_amount": 256369,
                "channel": "Mobile",
                "timestamp": "2025-08-12T02:10:24",
            }
        }


class PredictionOutput(BaseModel):
    is_fraud_prediction: int
    fraud_probability: float
    decision_threshold: float