from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Vocabulary Retention Predictor", layout="wide")
st.title("Vocabulary Retention Predictor")
st.caption("MVP dashboard: subject risk ranking and review workload outlook")

GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models/artifacts")

subject_features_path = GOLD_DIR / "subject_features_daily.csv"
risk_labels_path = GOLD_DIR / "risk_labels_daily.csv"
workload_features_path = GOLD_DIR / "workload_features_hourly.csv"

if not subject_features_path.exists() or not risk_labels_path.exists() or not workload_features_path.exists():
    st.warning("Missing gold tables. Run pipeline scripts first.")
    st.code(
        "\n".join(
            [
                "python src/pipeline/build_silver_tables.py",
                "python src/features/build_subject_features.py",
                "python src/features/build_workload_features.py",
                "python src/models/train_risk_model.py",
                "python src/models/train_workload_model.py",
            ]
        )
    )
    st.stop()

subject_features = pd.read_csv(subject_features_path)
risk_labels = pd.read_csv(risk_labels_path)
workload_features = pd.read_csv(workload_features_path)

risk_df = subject_features.merge(risk_labels, on=["snapshot_date", "subject_id"], how="left")

st.subheader("High-Risk Subjects")
subject_types = sorted(risk_df["subject_type"].dropna().unique().tolist()) if "subject_type" in risk_df.columns else []
selected_type = st.selectbox("Filter by subject type", options=["all"] + subject_types)

view = risk_df.copy()
if selected_type != "all":
    view = view[view["subject_type"] == selected_type]

if "risk_label_binary" in view.columns:
    view = view.sort_values(["risk_label_binary", "pct_correct"], ascending=[False, True])

cols_to_show = [
    "subject_id", "subject_type", "data_level", "data_srs_stage", "pct_correct",
    "total_attempts_est", "total_incorrect", "risk_label_multiclass", "risk_label_binary",
]
cols_to_show = [c for c in cols_to_show if c in view.columns]
st.dataframe(view[cols_to_show].head(50), use_container_width=True)

st.subheader("Workload Forecast View")
workload_features["hour_ts"] = pd.to_datetime(workload_features["hour_ts"], utc=True, errors="coerce")
plot_cols = [c for c in ["reviews_due_now", "reviews_due_next_1h", "reviews_due_next_3h", "reviews_due_next_24h"] if c in workload_features.columns]
if plot_cols:
    st.line_chart(workload_features.set_index("hour_ts")[plot_cols])

left, right = st.columns(2)
with left:
    st.metric("Rows in subject features", len(subject_features))
    st.metric("Rows in workload features", len(workload_features))

with right:
    risk_metrics_path = MODEL_DIR / "risk_model_metrics.csv"
    workload_metrics_path = MODEL_DIR / "workload_model_metrics.csv"
    if risk_metrics_path.exists():
        risk_metrics = pd.read_csv(risk_metrics_path).iloc[0].to_dict()
        st.write("Risk model metrics")
        st.json(risk_metrics)
    if workload_metrics_path.exists():
        workload_metrics = pd.read_csv(workload_metrics_path).iloc[0].to_dict()
        st.write("Workload model metrics")
        st.json(workload_metrics)
