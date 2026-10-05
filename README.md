# DDM501 Individual Assignments

Individual Assignment 1 and Individual Assignment 2 for **MSA36HN_DDM501**.

- **Student:** Nguyễn Thái Thịnh — 25MS13304
- **Course:** DDM501 — AI in DevOps, DataOps, MLOps

## Submission files

| Assignment | Report source | Submission PDF | Code and evidence |
| --- | --- | --- | --- |
| 1 — ML System Design | [Assignment1 report](Assignment1/DDM501_Assignment1_ML_System_Design.md) | [Assignment1 PDF](Assignment1/DDM501_Assignment1_25MS13304_Nguyen_Thai_Thinh.pdf) | Pydantic contracts, heuristic reference engine, focused tests, and evidence/limitations checklist |
| 2 — ML Pipeline Design & MLOps | [Assignment2 report](Assignment2/DDM501_Assignment2_ML_Pipeline_Design.md) | [Assignment2 PDF](Assignment2/DDM501_Assignment2_25MS13304_Nguyen_Thai_Thinh.pdf) | scikit-learn pipeline, ten-run synthetic benchmark, metadata, tests, optional Airflow prototype, and evidence/limitations checklist |

Each assignment is independently reproducible from its folder with `uv sync` and `uv run pytest -q`. Assignment 2's experiment matrix can be regenerated with `uv run python experiments.py`.

## Evidence boundary

Assignment 1's system figures are design assumptions and proposed targets. Its Python engine is a hand-written heuristic—not a trained or calibrated model.

Assignment 2 uses synthetic rows matching the UCI feature schema, not observations from the UCI dataset. The recorded experiment is one stratified 80/20 holdout; no candidate met the configured ROC-AUC gate, and none is promoted. MLflow tracking, DVC/MinIO, production Airflow, serving, and monitoring are proposed integrations, not deployed services.
