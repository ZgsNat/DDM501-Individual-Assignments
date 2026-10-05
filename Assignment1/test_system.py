"""
Unit and Contract Tests for Assignment 1 System Specification
"""

import pytest
import time
from pydantic import ValidationError
from system_spec import CreditApplicantInput, CreditRiskScoringEngine


def sample_applicant_payload(**overrides):
    base = {
        "LIMIT_BAL": 200000.0,
        "SEX": 2,
        "EDUCATION": 1,
        "MARRIAGE": 2,
        "AGE": 32,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 15000.0,
        "BILL_AMT2": 14000.0,
        "BILL_AMT3": 12000.0,
        "BILL_AMT4": 10000.0,
        "BILL_AMT5": 8000.0,
        "BILL_AMT6": 5000.0,
        "PAY_AMT1": 5000.0,
        "PAY_AMT2": 4000.0,
        "PAY_AMT3": 3000.0,
        "PAY_AMT4": 2000.0,
        "PAY_AMT5": 2000.0,
        "PAY_AMT6": 2000.0,
    }
    base.update(overrides)
    return base


def test_valid_applicant_schema():
    payload = sample_applicant_payload()
    applicant = CreditApplicantInput(**payload)
    assert applicant.LIMIT_BAL == 200000.0
    assert applicant.AGE == 32


def test_invalid_applicant_age_boundary():
    payload = sample_applicant_payload(AGE=15)  # Underage borrower
    with pytest.raises(ValidationError):
        CreditApplicantInput(**payload)


def test_negative_limit_balance():
    payload = sample_applicant_payload(LIMIT_BAL=-5000.0)
    with pytest.raises(ValidationError):
        CreditApplicantInput(**payload)


def test_prime_applicant_decision():
    engine = CreditRiskScoringEngine()
    applicant = CreditApplicantInput(**sample_applicant_payload(
        PAY_0=-1, PAY_2=-1, BILL_AMT1=5000.0, LIMIT_BAL=300000.0, PAY_AMT1=5000.0
    ))
    res = engine.evaluate(applicant)
    assert res.prediction == 0
    assert res.credit_tier in ["PRIME", "NEAR_PRIME"]
    assert res.risk_score >= 700
    assert res.decision == "approve"
    assert len(res.adverse_action_reasons) == 0


def test_subprime_applicant_is_routed_to_review_not_decline():
    engine = CreditRiskScoringEngine()
    applicant = CreditApplicantInput(**sample_applicant_payload(
        PAY_0=1,
        BILL_AMT1=180000.0,
        LIMIT_BAL=200000.0,
        PAY_AMT1=100.0,
        BILL_AMT2=10000.0,
    ))
    res = engine.evaluate(applicant)
    assert res.credit_tier == "SUBPRIME"
    assert res.decision == "review"
    assert res.adverse_action_reasons == []


def test_delinquent_applicant_adverse_action():
    engine = CreditRiskScoringEngine()
    applicant = CreditApplicantInput(**sample_applicant_payload(
        PAY_0=3, PAY_2=2, BILL_AMT1=190000.0, LIMIT_BAL=200000.0, PAY_AMT1=100.0
    ))
    res = engine.evaluate(applicant)
    assert res.prediction == 1
    assert res.credit_tier in ["SUBPRIME", "HIGH_RISK"]
    assert res.decision == "decline"
    assert len(res.adverse_action_reasons) > 0
    assert any("Delinquency" in r for r in res.adverse_action_reasons)


def test_local_heuristic_call_p95_under_50ms():
    engine = CreditRiskScoringEngine()
    applicant = CreditApplicantInput(**sample_applicant_payload())
    
    # Warmup
    for _ in range(5):
        engine.evaluate(applicant)
        
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        res = engine.evaluate(applicant)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95 < 50.0, f"local heuristic p95 {p95}ms exceeds the reference threshold"
