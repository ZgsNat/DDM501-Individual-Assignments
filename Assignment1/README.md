# DDM501 Individual Assignment 1

This folder contains the ML system-design report, a small Python contract/heuristic prototype, and focused tests. The architecture, business values, and service targets in the report are proposals, not evidence of a deployed bank system.

## Contents

- `DDM501_Assignment1_ML_System_Design.md` — report source
- `system_spec.py` — Pydantic input/output contracts and illustrative rule-based heuristic
- `test_system.py` — contract and behavior tests
- `SELF_EVALUATION.md` — rubric-to-evidence checklist and limitations
- `generate_pdf.py` — HTML/PDF report generator

## Run tests

```bash
uv sync
uv run pytest -q test_system.py
```

## Build the report

Install Google Chrome/Chromium or set `CHROME_BIN`, then build the named PDF:

```bash
uv run python generate_pdf.py --student-name "Full Name" --student-id "StudentID"
```

The script replaces the report's student name/ID and creates the required filename. The scoring heuristic is not a trained or calibrated model, a FICO score, a legally validated adverse-action system, or an end-to-end latency benchmark. The tests validate only the local software behavior they exercise.
