# DDM501 Individual Assignment 2

This assignment contains a local scikit-learn training/evaluation prototype, a ten-configuration benchmark, an optional Airflow DAG prototype, and a report distinguishing implemented behavior from proposed production integrations.

## Quickstart

```bash
uv sync
uv run pytest -q
uv run python experiments.py
```

The experiment command writes:

- `experiment_results.csv` and `experiment_results.json` — the ten-run comparison
- `experiment_metadata.json` — data source, split, thresholds, package versions, platform, and timing method

## Data source

With no configured path, the pipeline generates 5,000 synthetic rows matching the UCI credit dataset's feature schema. These are not UCI observations and are not representative evidence of real credit risk. Set `DATA_PATH=/path/to/your.csv` to use a CSV with the required feature columns and a `default` target (common UCI target aliases are normalized). If a configured path is missing, the pipeline fails explicitly rather than substituting synthetic data.

## Optional integrations

- `config.yaml` is loaded with `APP_ENV`, `DATA_PATH`, and `MLFLOW_TRACKING_URI` environment overrides.
- Airflow is not a project dependency. `airflow_dag.py` exposes a weekly candidate-evaluation DAG only when Airflow is installed. It does not connect to production data, MLflow, Evidently, or a serving API.
- The MLflow integration and registry/serving flow in the report are design examples, not executed runs.

The recorded benchmark is one stratified 80/20 holdout. All ten configurations failed the configured ROC-AUC eligibility threshold in the recorded synthetic run, so the report does not recommend promoting a model. Local inference timings cover only warmed-up model calls and exclude preprocessing and service/network overhead.

## Build the named PDF report

From the repository root, install the pinned local rendering assets with `npm ci`. Install Google Chrome/Chromium or set `CHROME_BIN`, then run:

```bash
uv run python generate_pdf.py --student-name "Full Name" --student-id "StudentID"
```

The command fills the report's author fields, renders LaTeX equations and Mermaid flowcharts, verifies the browser output, and generates the assignment's required PDF filename. It fails explicitly if the browser is not found, an equation/diagram fails to render, or PDF generation fails.
