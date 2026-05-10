"""Optimization studies — convergence, best params, trial table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.dashboard._paths import list_study_dirs, read_json

st.set_page_config(page_title="Optimization Studies", page_icon="🎯", layout="wide")
st.title("Optimization Studies")

studies = list_study_dirs()
if not studies:
    st.warning("No studies under `data/studies/`. Run one with `scripts.optimize_deployment`.")
    st.stop()

choice = st.selectbox(
    "Select a study", options=range(len(studies)),
    format_func=lambda i: studies[i].name,
)
sd: Path = studies[choice]

st.caption(f"`{sd}`")

study_cfg = read_json(sd / "study.json")
best = read_json(sd / "best.json")

c1, c2, c3, c4 = st.columns(4)
c1.metric("optimizer", best.get("optimizer", "—"))
c2.metric("# trials", best.get("n_trials", "—"))
c3.metric("best objective",
          f"{best.get('objective'):.3f}" if best.get("objective") is not None else "—")
c4.metric("direction", best.get("direction", "—"))

st.divider()

conv = sd / "plot_convergence.png"
if conv.exists():
    st.subheader("Convergence")
    st.image(str(conv), use_container_width=True)

st.divider()

cl, cr = st.columns(2)
with cl:
    st.subheader("Best parameters")
    if best.get("params"):
        st.json(best["params"])
    st.subheader("Best KPIs")
    if best.get("kpis"):
        st.json(best["kpis"])

with cr:
    st.subheader("Threat / objective")
    if study_cfg.get("threat"):
        st.json(study_cfg["threat"])
    if study_cfg.get("objective"):
        st.json(study_cfg["objective"])

st.divider()

st.subheader("Trials")
trials_path = sd / "trials.parquet"
if trials_path.exists():
    df = pd.read_parquet(trials_path)
    st.dataframe(df, use_container_width=True)

    kpi_cols = [c for c in df.columns if c.startswith("kpi.") and pd.api.types.is_numeric_dtype(df[c])]
    if "objective" in df.columns and kpi_cols:
        st.divider()
        st.subheader("Trial scatter")
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X", ["trial", "objective"] + kpi_cols, index=0)
        with c2:
            y = st.selectbox("Y", ["objective"] + kpi_cols, index=0)
        st.scatter_chart(df, x=x, y=y)
