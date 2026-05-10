"""Wireless-Sec-Sim — Streamlit dashboard home.

Run with:
    streamlit run services/dashboard/Home.py
"""

from __future__ import annotations

import streamlit as st

from services.dashboard._paths import (
    list_dataset_dirs,
    list_embedder_dirs,
    list_model_dirs,
    list_single_runs,
    list_study_dirs,
    list_sweep_dirs,
    vector_store_path,
)

st.set_page_config(
    page_title="Wireless-Sec-Sim",
    page_icon="📡",
    layout="wide",
)

st.title("Wireless-Sec-Sim")
st.caption(
    "AI-driven secure wireless network design and optimization platform "
    "— interactive dashboard."
)

single_runs = list_single_runs()
sweeps = list_sweep_dirs()
studies = list_study_dirs()
datasets = list_dataset_dirs()
models = list_model_dirs()
embedders = list_embedder_dirs()

cols = st.columns(6)
cols[0].metric("single runs", len(single_runs))
cols[1].metric("sweeps", len(sweeps))
cols[2].metric("studies", len(studies))
cols[3].metric("datasets", len(datasets))
cols[4].metric("models", len(models))
cols[5].metric("embedders", len(embedders))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Single runs (Phase 2)")
    if single_runs:
        for r in single_runs:
            st.write(f"`{r.relative_to(r.parents[1])}`")
    else:
        st.info("No `data/runs/` outputs yet. Try `python -m scripts.run_scenario configs/munich_baseline.yaml`.")

    st.subheader("Sweeps (Phase 3)")
    if sweeps:
        for s in sweeps:
            n_runs = len(list((s / 'runs').glob('*'))) if (s / 'runs').exists() else 0
            st.write(f"`{s.name}` — {n_runs} runs")
    else:
        st.info("No `data/sweeps/` outputs yet. Try `python -m scripts.run_sweep configs/sweeps/jammer_position.yaml`.")

    st.subheader("Studies (Phase 6)")
    if studies:
        for st_dir in studies:
            st.write(f"`{st_dir.name}`")
    else:
        st.info("No `data/studies/` outputs yet. Try `python -m scripts.optimize_deployment configs/studies/munich_bs_perf.yaml`.")

with right:
    st.subheader("Datasets (Phase 4)")
    if datasets:
        for d in datasets:
            st.write(f"`{d.name}`")
    else:
        st.info("No datasets yet.")

    st.subheader("Models (Phase 4)")
    if models:
        for m in models:
            st.write(f"`{m.name}`")
    else:
        st.info("No models yet.")

    st.subheader("Embedders & vector store (Phase 5)")
    if embedders:
        for e in embedders:
            st.write(f"`{e.name}`")
    else:
        st.info("No embedders yet.")
    vs = vector_store_path()
    if vs:
        st.success(f"Vector store: `{vs}`")
    else:
        st.warning("No vector store at `data/vector_store/`.")

st.divider()

st.markdown(
    """
    ### Pages

    - **Run Browser** — pick any per-run directory and inspect its 3D views, KPIs, and per-link table.
    - **Sweep Results** — pick a sweep and see its summary table + attack-zone heatmaps.
    - **Similar Scenarios** — query the vector store with a run and see its k nearest neighbours side-by-side.
    - **Optimization Studies** — pick a study and see its convergence trace + best parameters + KPIs.
    """
)
