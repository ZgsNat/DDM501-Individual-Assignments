# Assignment 2: Evidence and Rubric Alignment

This checklist is an evidence map, not a self-awarded score. The instructor determines the final grade.

| Rubric area | Evidence in submission | Scope / evidence limitation |
| --- | --- | --- |
| Pipeline design (25%) | Local pipeline code, validation gates, preprocessing, training/evaluation flow, and proposed production diagram | Only local training/evaluation is implemented; database ingestion, delayed labels, model registry, and serving are not |
| Experiment tracking and metrics (25%) | Ten executed scikit-learn configurations; CSV/JSON outputs and environment metadata; metrics and trade-off analysis | One 80/20 holdout on generated synthetic rows, not UCI data, cross-validation, MLflow tracking, or real credit performance |
| Workflow orchestration (20%) | Optional Airflow DAG prototype with real pipeline callables and candidate gate; target DAG and trigger design | Airflow is optional and was not executed in this environment; drift, alerts, MLflow promotion, and serving reload are future work |
| Code quality and documentation (20%) | Typed interfaces/docstrings, configuration models, data checks, and 15 focused tests | No lint/CI compliance claim; tests do not establish production reliability or legal compliance |
| Reproducibility/versioning (10%) | Fixed synthetic-data seed, estimator/split seeds, experiment metadata, Git-based source control | DVC/MinIO/MLflow model versioning and Docker deployment are proposed, not configured |

## Measured experiment outcome

The recorded synthetic-data run produced no candidate meeting the configured minimum ROC-AUC gate of 0.75. Accordingly, all non-baseline configurations are marked rejected and no champion is recommended. The DIR values are a limited two-group screening statistic at a fixed threshold, not proof of fairness or legal compliance.

## Remaining evidence needed

Repeat the matrix against a documented, representative dataset with a predefined split/CV protocol; record data and environment versions; evaluate calibration, subgroup and intersectional fairness, and decision costs; then test actual MLflow/Airflow/serving integrations before making deployment claims.
