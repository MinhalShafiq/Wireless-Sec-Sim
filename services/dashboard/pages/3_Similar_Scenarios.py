"""Retrieval-augmented decision support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.dashboard._paths import (
    all_runs_under_sweeps_and_singles,
    list_embedder_dirs,
    vector_store_path,
)

st.set_page_config(page_title="Similar Scenarios", page_icon="🔎", layout="wide")
st.title("Similar Scenarios")

vs = vector_store_path()
embedders = list_embedder_dirs()
if not vs or not embedders:
    st.warning("No vector store or embedder. Run `python -m scripts.ingest_runs` first.")
    st.stop()

embedder_dir = st.selectbox(
    "Embedder", options=embedders, format_func=lambda p: p.name,
)

runs = all_runs_under_sweeps_and_singles()
if not runs:
    st.warning("No runs available to query.")
    st.stop()

labels = [str(r.relative_to(r.parents[2])) for r in runs]
sel = st.selectbox(
    "Query run", options=range(len(runs)), format_func=lambda i: labels[i],
)
query_run: Path = runs[sel]

c1, c2 = st.columns([1, 1])
with c1:
    k = st.slider("k", min_value=1, max_value=20, value=5)
with c2:
    label_filter = st.selectbox(
        "Filter by label",
        options=["any", "benign (0)", "attack (1)"],
    )

filters: dict | None = None
if label_filter == "benign (0)":
    filters = {"label": 0}
elif label_filter == "attack (1)":
    filters = {"label": 1}

with st.spinner("Embedding + querying…"):
    from libs.retrieval import ScenarioEmbedder, ScenarioStore, summarize_similar

    embedder = ScenarioEmbedder.load(embedder_dir)
    store = ScenarioStore(path=str(vs), dim=embedder.dim)
    summary = summarize_similar(query_run, embedder, store, k=k, filters=filters)

# --- Header metrics ---------------------------------------------------------
ns = summary["neighbour_summary"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("k neighbours", ns.get("k", 0))
share = ns.get("share_attack")
m2.metric("share attack", f"{share:.0%}" if isinstance(share, (int, float)) else "—")
median_sinr = ns.get("median_sinr_db_across_neighbors")
m3.metric("neighbour median SINR (dB)",
          f"{median_sinr:.1f}" if isinstance(median_sinr, (int, float)) else "—")
m4.metric("any rogue / jammer",
          ("rogue" if ns.get("any_rogue") else "") + (" jammer" if ns.get("any_jammer") else "") or "no")

st.divider()

# --- Match list -------------------------------------------------------------
st.subheader("Top matches")
matches_df = pd.DataFrame(summary["matches"])
if not matches_df.empty:
    st.dataframe(matches_df, use_container_width=True)
else:
    st.info("No matches returned.")

# --- Side-by-side 3D views: query vs top match ------------------------------
top = summary["matches"][0] if summary["matches"] else None
if top:
    st.divider()
    st.subheader("Query vs top match — 3D views")
    # Find top match's run dir
    top_run_id = top["run_id"]
    top_run_dir = next(
        (r for r in all_runs_under_sweeps_and_singles() if r.name == top_run_id),
        None,
    )
    cols = st.columns(2)
    for col, label, rdir in (
        (cols[0], f"QUERY: {query_run.name}", query_run),
        (cols[1], f"MATCH: {top_run_id}", top_run_dir),
    ):
        with col:
            st.markdown(f"**{label}**")
            if rdir is None:
                st.warning("Match run dir not found on disk.")
                continue
            for v in ("view_overview.png", "view_sinr.png"):
                vp = rdir / v
                if vp.exists():
                    st.image(str(vp), caption=v, use_container_width=True)
