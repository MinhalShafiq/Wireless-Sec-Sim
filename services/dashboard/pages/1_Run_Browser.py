"""Run browser — inspect a single per-run directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.dashboard._paths import (
    all_runs_under_sweeps_and_singles,
    read_json,
)

st.set_page_config(page_title="Run Browser", page_icon="📡", layout="wide")
st.title("Run Browser")

runs = all_runs_under_sweeps_and_singles()
if not runs:
    st.warning("No runs found. Run a scenario or a sweep first.")
    st.stop()

labels = [str(r.relative_to(r.parents[2])) for r in runs]
choice = st.selectbox("Select a run", options=range(len(runs)), format_func=lambda i: labels[i])
run_dir: Path = runs[choice]

st.caption(f"`{run_dir}`")
st.divider()

# --- KPIs --------------------------------------------------------------------
kpis = read_json(run_dir / "kpis.json")
agg = kpis.get("aggregate", {})
per_rx = kpis.get("per_rx_summary", [])

mcols = st.columns(5)
mcols[0].metric("# users", agg.get("num_users", "—"))
mcols[1].metric("# eavesdroppers", agg.get("num_eavesdroppers", "—"))
sinr = agg.get("median_user_sinr_db")
mcols[2].metric("median UE SINR (dB)", f"{sinr:.1f}" if isinstance(sinr, (int, float)) else "—")
mcols[3].metric("rogue-dominated", agg.get("users_with_rogue_dominant", "—"))
mcols[4].metric("jammer-dominated", agg.get("users_with_jammer_dominant", "—"))

st.divider()

# --- 3D views ---------------------------------------------------------------
st.subheader("3D views")
view_files = sorted(p for p in run_dir.glob("view_*.png"))
if view_files:
    img_cols = st.columns(min(3, len(view_files)) or 1)
    for i, vf in enumerate(view_files):
        with img_cols[i % len(img_cols)]:
            st.image(str(vf), caption=vf.name, use_container_width=True)
else:
    st.info("No 3D views in this run. Re-run with rendering enabled (omit `--no-render`).")

# 2D matplotlib radio map (always emitted)
rm_png = run_dir / "radio_map.png"
if rm_png.exists():
    st.subheader("2D radio map (TX 0)")
    st.image(str(rm_png), use_container_width=False)

st.divider()

# --- Tabs for the rest ------------------------------------------------------
tab_kpi, tab_links, tab_scenario, tab_meta = st.tabs(
    ["Per-RX KPIs", "Per-link table", "Scenario", "Meta"]
)

with tab_kpi:
    if per_rx:
        st.dataframe(pd.DataFrame(per_rx), use_container_width=True)
    else:
        st.info("No per-RX summary in kpis.json.")

with tab_links:
    parquet = run_dir / "per_link.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        st.caption(f"{len(df)} per-(RX, TX) rows.")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No per_link.parquet in this run.")

with tab_scenario:
    scen = read_json(run_dir / "scenario.json")
    if scen:
        st.json(scen)

with tab_meta:
    meta = read_json(run_dir / "meta.json")
    if meta:
        st.json(meta)
