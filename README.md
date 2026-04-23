# Vocabulary-Retention-Predictor

MVP scaffold for a portfolio project that predicts subject-level retention risk and short-horizon review workload using WaniKani API exports.

## What is implemented

- Silver table builder from raw endpoint exports
- Gold feature builders for subject risk and workload forecasting
- Baseline training scripts for:
	- binary subject risk classification
	- next-hour workload regression
- Streamlit dashboard page for high-risk subject ranking and workload charts

## Project structure

- `config/project_config.yaml`
- `src/pipeline/build_silver_tables.py`
- `src/features/build_subject_features.py`
- `src/features/build_workload_features.py`
- `src/models/train_risk_model.py`
- `src/models/train_workload_model.py`
- `app/dashboard.py`

## Data flow

1. Raw JSON files in `data/raw`
2. Silver normalized CSVs in `data/silver`
3. Gold model-ready CSVs in `data/gold`
4. Trained artifacts and metrics in `models/artifacts`

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Build silver tables:

```bash
python src/pipeline/build_silver_tables.py
```

Build gold features:

```bash
python src/features/build_subject_features.py
python src/features/build_workload_features.py
```

Train baseline models:

```bash
python src/models/train_risk_model.py
python src/models/train_workload_model.py
```

Run dashboard:

```bash
streamlit run app/dashboard.py
```

## Notes

- The historical `reviews` retrieval endpoint is deprecated, so this scaffold is centered on `assignments`, `review_statistics`, `subjects`, and `summary`.
- Current labels in `build_subject_features.py` are weak-supervision rules suitable for MVP iteration.
- SRS stage 0 subjects (locked, not yet started) are excluded from gold features, risk labels, model training, and dashboard ranking. They have no review history and would otherwise dominate high-risk outputs due to the `srs_stage <= 3` labeling rule. The exclusion is applied in `build_subject_features.py` before label creation, with a defensive guardrail also present in `dashboard.py`. Silver and raw data are unaffected. Workload forecasting is also unaffected as it derives from hourly summary data, not subject SRS stage.

