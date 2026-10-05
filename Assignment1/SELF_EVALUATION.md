# Assignment 1: Evidence and Rubric Alignment

This checklist maps the report and prototype to the assignment criteria. It is not a predicted grade; the instructor determines the score.

| Rubric area | Evidence in submission | Scope / evidence limitation |
| --- | --- | --- |
| Problem definition (20%) | Domain problem statement, scenario, stakeholder matrix, and rationale for ML | Business volumes, costs, manual-review rates, and portfolio outcomes are assumptions, not measured facts |
| Requirements analysis (20%) | Functional/non-functional requirements, feature schema, and data governance considerations | Requirements are proposed; no bank, serving API, database, or compliance review is included |
| Goals and metrics (20%) | Business/system/model metric hierarchy and target thresholds | Targets are not achieved results; model metrics require representative data and a defined evaluation protocol |
| High-level architecture (25%) | Data-flow and subsystem diagrams covering serving, telemetry, monitoring, and retraining | Diagram is a design proposal; listed services are not deployed in this assignment |
| Trade-offs (15%) | Discussion of model complexity, freshness, interpretability, and oversight | Quantitative comparisons are explicitly not claimed without a reproducible benchmark |

## Runnable evidence

- `system_spec.py` implements Pydantic input/output contracts and a hand-written heuristic.
- `test_system.py` verifies schema boundaries, example decision bands, explanations, and local function timing.
- The heuristic is not a trained model, calibrated probability estimator, FICO score, legally validated adverse-action engine, or production service.
- The timing test excludes API, database, network, and deployment overhead.

## Remaining evidence needed for a production claim

Use a lawfully obtained representative dataset, document its provenance, run leakage-safe validation, evaluate calibration and subgroup/intersectional fairness, conduct legal/model-risk review, and test the complete service under load. Do not treat the scenario targets or the local unit tests as evidence that these outcomes have been achieved.
