"""Sweep-level visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_2d_heatmap(
    df: pd.DataFrame, x_col: str, y_col: str, metric_col: str, out_path: Path, *,
    title: str | None = None, cmap: str = "viridis",
):
    """Heatmap of ``metric_col`` over the (x_col, y_col) grid.

    Averages over any other columns (e.g., seed) for the same (x,y) cell.
    """
    pivot = df.groupby([y_col, x_col])[metric_col].mean().unstack(x_col)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        pivot.values, origin="lower", cmap=cmap, aspect="auto",
        extent=[
            float(pivot.columns.min()), float(pivot.columns.max()),
            float(pivot.index.min()),   float(pivot.index.max()),
        ],
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{metric_col} over {x_col} × {y_col}")
    fig.colorbar(im, ax=ax, label=metric_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_1d_line(df: pd.DataFrame, x_col: str, metric_col: str, out_path: Path, *,
                 title: str | None = None):
    grouped = df.groupby(x_col)[metric_col].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(grouped[x_col], grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric_col)
    ax.set_title(title or f"{metric_col} vs {x_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def auto_plots(summary_path: Path) -> list[Path]:
    """Inspect summary.parquet and emit sensible default plots.

    - For sweeps with both ``param.position_x`` and ``param.position_y``,
      produce a 2D heatmap of each numeric KPI.
    - For 1D sweeps (single grid axis), produce per-KPI line plots.
    """
    df = pd.read_parquet(summary_path)
    out_dir = summary_path.parent
    plots: list[Path] = []

    param_cols = [c for c in df.columns if c.startswith("param.")]
    kpi_cols = [c for c in df.columns
                if c.startswith("kpi.") and pd.api.types.is_numeric_dtype(df[c])]

    if "param.position_x" in param_cols and "param.position_y" in param_cols:
        for kc in kpi_cols:
            metric = kc.removeprefix("kpi.")
            out = out_dir / f"plot_xy_{metric}.png"
            try:
                plot_2d_heatmap(
                    df, "param.position_x", "param.position_y", kc, out,
                    title=f"{metric} over jammer/rogue (x, y)",
                )
                plots.append(out)
            except Exception as e:  # noqa: BLE001
                print(f"[plot] skipped {kc}: {e}")
    elif len(param_cols) == 1:
        x = param_cols[0]
        for kc in kpi_cols:
            metric = kc.removeprefix("kpi.")
            out = out_dir / f"plot_{x.removeprefix('param.')}_vs_{metric}.png"
            try:
                plot_1d_line(df, x, kc, out)
                plots.append(out)
            except Exception as e:  # noqa: BLE001
                print(f"[plot] skipped {kc}: {e}")

    # Multi-seed scatter / histogram of median SINR (if present)
    if "kpi.median_user_sinr_db" in df.columns and len(df) > 1:
        out = out_dir / "plot_sinr_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["kpi.median_user_sinr_db"].dropna(), bins=20, edgecolor="black")
        ax.set_xlabel("median user SINR (dB)")
        ax.set_ylabel("# runs")
        ax.set_title("SINR distribution across sweep")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plots.append(out)

    return plots
