"""Airflow prototype for scheduled training and candidate quality-gate evaluation."""

from datetime import datetime, timedelta
from typing import Any

try:
    from airflow import DAG
    from airflow.exceptions import AirflowFailException
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None
    AirflowFailException = RuntimeError


DEFAULT_ARGS = {
    "owner": "ddm501-assignment",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


def task_train_and_evaluate(**context: Any) -> dict[str, Any]:
    """Run the actual local pipeline and return only its compact metrics dictionary."""
    from pipeline import CreditRiskPipeline

    return CreditRiskPipeline().run_pipeline()


def task_candidate_quality_gate(**context: Any) -> dict[str, Any]:
    """Fail the run unless the measured candidate passes the configured offline gates."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="train_and_evaluate_candidate")
    if not isinstance(metrics, dict):
        raise AirflowFailException("Training task did not return a metrics dictionary.")
    if not metrics.get("passed_quality_gates", False):
        raise AirflowFailException(
            "Candidate did not pass configured ROC-AUC and disparate-impact gates; "
            "no model is promoted."
        )
    return {"status": "eligible_for_manual_review", "metrics": metrics}


def create_dag():
    """Create the prototype DAG when Airflow is installed; otherwise expose no fake DAG."""
    if not AIRFLOW_AVAILABLE:
        return None

    with DAG(
        dag_id="credit_risk_candidate_evaluation",
        default_args=DEFAULT_ARGS,
        description="Train and evaluate a candidate; does not register or deploy models.",
        schedule="@weekly",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["mlops", "credit-risk", "assignment"],
    ) as dag:
        train = PythonOperator(
            task_id="train_and_evaluate_candidate",
            python_callable=task_train_and_evaluate,
        )
        gate = PythonOperator(
            task_id="validate_candidate_quality_gates",
            python_callable=task_candidate_quality_gate,
        )
        train >> gate
        return dag


dag = create_dag()
