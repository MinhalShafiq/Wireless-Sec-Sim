"""Sweep results — summary table, attack-zone heatmaps, distribution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.dashboard._paths import list_sweep_dirs, read_json

st.set_page_config(page_title="Sweep Results", page_icon="📊", layout="wide")
st.title("Sweep Results")

sweeps = list_sweep_dirs()
if not sweeps:
    st.warning("No sweeps under `data/sweeps/`. Run one with `scripts.run_sweep`.")
    st.stop()

choice = st.selectbox(
    "Select a sweep", options=range(len(sweeps)),
    format_func=lambda i: sweeps[i].name,
)
sd: Path = sweeps[choice]

st.caption(f"`{sd}`")

# --- Header metrics ---------------------------------------------------------
sweep_cfg = read_json(sd / "sweep.json")
summary = read_json(sd / "summary.json")
kc1, kc2, kc3, kc4 = st.columns(4)
kc1.metric("# runs", summary.get("n_runs", "—"))
kc2.metric("optimizer/template", sweep_cfg.get("template", "—"))
kc3.metric("total time (s)", summary.get("total_time_s", "—"))
kc4.metric("mean s/run", summary.get("mean_time_per_run_s", "—"))

st.divider()

# --- Attack-zone heatmaps ---------------------------------------------------
st.subheader("Attack-zone heatmaps")
xy_plots = sorted(sd.glob("plot_xy_*.png"))
if xy_plots:
    img_cols = st.columns(min(2, len(xy_plots)) or 1)
    for i, p in enumerate(xy_plots):
        with img_cols[i % len(img_cols)]:
            st.image(str(p), caption=p.stem.replace("plot_xy_", ""), use_container_width=True)
else:
    st.info("No XY heatmaps in this sweep.")

dist = sd / "plot_sinr_distribution.png"
if dist.exists():
    st.subheader("SINR distribution across runs")
    st.image(str(dist), use_container_width=False)

st.divider()

# --- Summary table ----------------------------------------------------------
st.subheader("Summary parquet")
parq = sd / "summary.parquet"
if parq.exists():
    df = pd.read_parquet(parq)
    st.caption(f"{len(df)} rows × {df.shape[1]} cols.")
    st.dataframe(df, use_container_width=True)

    # Quick scatter of any numeric KPI vs first numeric param.
    kpi_cols = [c for c in df.columns if c.startswith("kpi.") and pd.api.types.is_numeric_dtype(df[c])]
    param_cols = [c for c in df.columns if c.startswith("param.")]
    if kpi_cols and param_cols:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            xc = st.selectbox("X (param.*)", param_cols, index=0)
        with c2:
            yc = st.selectbox("Y (kpi.*)", kpi_cols, index=0)
        st.scatter_chart(df, x=xc, y=yc)
