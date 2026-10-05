"""
Enterprise Credit Default Risk Scoring - System Specification & Verification Module
Course: DDM501 - AI in DevOps, DataOps, MLOps (Individual Assignment 1)
"""

from typing import List, Literal
import time
import math
from pydantic import BaseModel, Field, field_validator


class CreditApplicantInput(BaseModel):
    """
    Input schema for the 23 features of the UCI Credit Card Default Dataset.
    Enforces strict domain boundaries, types, and constraints.
    """
    LIMIT_BAL: float = Field(..., gt=0, description="Amount of given credit in NT dollars")
    SEX: int = Field(..., ge=1, le=2, description="1=Male, 2=Female")
    EDUCATION: int = Field(..., ge=1, le=4, description="1=Grad School, 2=University, 3=High School, 4=Others")
    MARRIAGE: int = Field(..., ge=1, le=3, description="1=Married, 2=Single, 3=Others")
    AGE: int = Field(..., ge=18, le=100, description="Age in years (18 to 100)")
    
    # Repayment status over past 6 months (-1=Pay duly, 0=Revolving, 1-9=Delay months)
    PAY_0: int = Field(..., ge=-2, le=9)
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    
    # Historical monthly bill statements (NT dollars)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    
    # Historical monthly payment settlement amounts (NT dollars)
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    @field_validator("AGE")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 18 or v > 100:
            raise ValueError("Age must be between 18 and 100")
        return v


class CreditPredictionResponse(BaseModel):
    """
    Contract for an illustrative heuristic response, not a deployed-model result.
    """
    prediction: int = Field(..., description="Heuristic risk flag: 0 below 0.30, otherwise 1")
    decision: Literal["approve", "review", "decline"]
    probability: float = Field(..., ge=0.0, le=1.0, description="Uncalibrated heuristic risk estimate")
    risk_score: int = Field(..., ge=300, le=850, description="Illustrative score; not a bureau/FICO score")
    credit_tier: str = Field(..., description="Risk tier: PRIME, NEAR_PRIME, SUBPRIME, HIGH_RISK")
    recommended_credit_limit: float = Field(..., description="Recommended dynamic credit line in NTD")
    adverse_action_reasons: List[str] = Field(
        default_factory=list,
        description="Heuristic explanation candidates, not approved legal reason codes",
    )
    latency_ms: float = Field(..., description="Local heuristic execution time; excludes API and infrastructure")


class CreditRiskScoringEngine:
    """
    Hand-written reference heuristic for illustrating input/output contracts and decision bands.

    It is not a trained or calibrated model, a regulatory decision engine, or an end-to-end
    latency benchmark.
    """

    def __init__(self, model_version: str = "heuristic-v1"):
        self.model_version = model_version

    def evaluate(self, applicant: CreditApplicantInput) -> CreditPredictionResponse:
        start_time = time.perf_counter()

        # Hand-written heuristic; this is not a fitted GBDT decision surface.
        utilization = applicant.BILL_AMT1 / max(applicant.LIMIT_BAL, 1.0)
        repayment_delinquency = max(applicant.PAY_0, applicant.PAY_2, applicant.PAY_3)
        payment_ratio = applicant.PAY_AMT1 / max(abs(applicant.BILL_AMT2), 1.0)

        # Baseline logit estimation
        logit = -2.20
        logit += 0.85 * max(0, repayment_delinquency)
        logit += 1.20 * min(2.0, utilization)
        if payment_ratio < 0.10:
            logit += 0.45
        if applicant.AGE < 25:
            logit += 0.20

        # Logistic transform to a bounded heuristic score, not statistical calibration.
        probability = 1.0 / (1.0 + math.exp(-logit))
        probability = max(0.001, min(0.999, round(probability, 4)))

        risk_score = int(850 - (probability * 550))
        risk_score = max(300, min(850, risk_score))

        # Illustrative decision tiers; production actions require validated policy.
        if probability < 0.15:
            credit_tier = "PRIME"
            prediction = 0
            decision = "approve"
            rec_limit = applicant.LIMIT_BAL * 1.15
        elif probability < 0.30:
            credit_tier = "NEAR_PRIME"
            prediction = 0
            decision = "approve"
            rec_limit = applicant.LIMIT_BAL * 1.00
        elif probability < 0.60:
            credit_tier = "SUBPRIME"
            prediction = 1
            decision = "review"
            rec_limit = applicant.LIMIT_BAL * 0.50
        else:
            credit_tier = "HIGH_RISK"
            prediction = 1
            decision = "decline"
            rec_limit = 0.0

        # Candidate explanations are emitted only for the illustrative decline band.
        adverse_reasons = []
        if decision == "decline":
            if repayment_delinquency > 0:
                adverse_reasons.append(f"Recent Delinquency Detected: Delay of {repayment_delinquency} months")
            if utilization > 0.70:
                adverse_reasons.append(f"High Credit Line Utilization: {utilization * 100:.1f}% of limit used")
            if payment_ratio < 0.20:
                adverse_reasons.append(f"Low Recent Settlement Ratio: Paid {payment_ratio * 100:.1f}% of bill")
            if not adverse_reasons:
                adverse_reasons.append("Elevated Aggregate Credit Risk Profile")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return CreditPredictionResponse(
            prediction=prediction,
            decision=decision,
            probability=probability,
            risk_score=risk_score,
            credit_tier=credit_tier,
            recommended_credit_limit=round(rec_limit, 2),
            adverse_action_reasons=adverse_reasons[:3],
            latency_ms=round(elapsed_ms, 2)
        )
