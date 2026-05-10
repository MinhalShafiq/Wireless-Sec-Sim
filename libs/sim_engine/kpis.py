"""KPI extraction from Sionna RT path-tracing output.

Given the ``Paths`` object from a :class:`sionna.rt.PathSolver`, this
module computes:

- **per-(RX, TX) path gain** (linear and dB) by summing |coef|^2 over
  valid paths,
- **per-RX best server** (strongest legitimate TX),
- **per-RX SINR** including all other TXs as interference plus thermal
  noise (with the configured noise figure),
- **attack flags**: was a rogue TX the strongest signal at the UE? did
  a jammer dominate the interference?

These map directly onto the security metrics promised in the design
proposal (best-server flips, jammer dominance, secrecy capacity inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from libs.schemas import Scenario

if TYPE_CHECKING:
    pass

BOLTZMANN = 1.380649e-23  # J/K
T_KELVIN = 290.0


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Thermal noise floor in dBm for the given bandwidth + NF."""
    n0_dbm = 10 * np.log10(BOLTZMANN * T_KELVIN * bandwidth_hz) + 30  # W → dBm
    return n0_dbm + noise_figure_db


def _paths_to_path_gain(paths) -> np.ndarray:
    """Reduce a ``Paths`` object to a per-(RX, TX) linear path-gain matrix.

    Sionna RT's ``paths.a`` is ``(re, im)`` tensors of shape
    ``(num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)``. We sum
    ``|coef|^2`` over the path axis after masking with ``paths.valid``
    (shape ``(num_rx, num_tx, num_paths)``).
    """
    re, im = paths.a
    re = np.asarray(re)
    im = np.asarray(im)
    # Squeeze antenna dims (we use 1×1 isotropic arrays in Phase 2).
    re_sq = re.squeeze(axis=(1, 3))
    im_sq = im.squeeze(axis=(1, 3))
    power_per_path = re_sq**2 + im_sq**2  # (num_rx, num_tx, num_paths)
    valid = np.asarray(paths.valid).astype(np.float32)  # (num_rx, num_tx, num_paths)
    power_per_path = power_per_path * valid
    return power_per_path.sum(axis=-1)  # (num_rx, num_tx)


@dataclass
class RxKpiRow:
    rx_name: str
    rx_role: str
    rx_xyz: tuple[float, float, float]
    tx_name: str
    tx_role: str
    tx_xyz: tuple[float, float, float]
    tx_power_dbm: float
    distance_m: float
    path_gain_db: float
    rx_power_dbm: float


@dataclass
class RxSummary:
    rx_name: str
    rx_role: str
    best_server_tx: str | None
    best_server_role: str | None
    best_server_rx_power_dbm: float | None
    interferer_power_dbm: float
    sinr_db: float | None
    rogue_dominates: bool
    jammer_dominates: bool


@dataclass
class RunKpis:
    per_link_rows: list[RxKpiRow]
    per_rx_summary: list[RxSummary]
    aggregate: dict


def compute_kpis(scenario: Scenario, paths) -> RunKpis:
    """Compute the full KPI bundle from a Paths object."""
    pg_lin = _paths_to_path_gain(paths)  # (num_rx, num_tx)
    num_rx, num_tx = pg_lin.shape

    txs = scenario.transmitters
    rxs = scenario.receivers
    assert num_tx == len(txs), f"path-gain TX dim {num_tx} != scenario TX count {len(txs)}"
    assert num_rx == len(rxs), f"path-gain RX dim {num_rx} != scenario RX count {len(rxs)}"

    tx_pwr_lin = np.array([10 ** (t.power_dbm / 10) * 1e-3 for t in txs])  # W
    rx_power_w = pg_lin * tx_pwr_lin[None, :]  # (num_rx, num_tx)

    n0_dbm = thermal_noise_dbm(scenario.band.bandwidth_hz, scenario.band.noise_figure_db)
    n0_w = 10 ** ((n0_dbm - 30) / 10)

    per_link_rows: list[RxKpiRow] = []
    per_rx_summary: list[RxSummary] = []

    for i, rx in enumerate(rxs):
        rx_xyz = (rx.position.x, rx.position.y, rx.position.z)
        for j, tx in enumerate(txs):
            tx_xyz = (tx.position.x, tx.position.y, tx.position.z)
            d = float(np.linalg.norm(np.array(rx_xyz) - np.array(tx_xyz)))
            pg = pg_lin[i, j]
            pg_db = 10 * np.log10(pg) if pg > 0 else float("-inf")
            rx_pow_dbm = tx.power_dbm + pg_db
            per_link_rows.append(
                RxKpiRow(
                    rx_name=rx.name, rx_role=rx.role, rx_xyz=rx_xyz,
                    tx_name=tx.name, tx_role=tx.role, tx_xyz=tx_xyz,
                    tx_power_dbm=tx.power_dbm, distance_m=d,
                    path_gain_db=pg_db, rx_power_dbm=rx_pow_dbm,
                )
            )

        # Per-RX summary
        rx_pow_row = rx_power_w[i, :]
        # Best legitimate server
        legit_mask = np.array([t.role == "legitimate" for t in txs])
        if legit_mask.any() and (rx_pow_row * legit_mask).sum() > 0:
            best_j = int(np.argmax(rx_pow_row * legit_mask))
            best_pow_w = float(rx_pow_row[best_j])
            best_tx = txs[best_j]
            best_name, best_role = best_tx.name, best_tx.role
            best_dbm: float | None = 10 * np.log10(best_pow_w) + 30
            interf_w = float(rx_pow_row.sum() - best_pow_w)
            sinr = best_pow_w / (interf_w + n0_w)
            sinr_db: float | None = 10 * np.log10(sinr) if sinr > 0 else None
        else:
            best_j = -1
            best_pow_w = 0.0
            best_name = best_role = None
            best_dbm = None
            interf_w = float(rx_pow_row.sum())
            sinr_db = None

        # Adversarial flags
        strongest_j = int(np.argmax(rx_pow_row)) if rx_pow_row.size else -1
        rogue_dominates = strongest_j >= 0 and txs[strongest_j].role == "rogue"
        jammer_powers = np.array(
            [rx_pow_row[k] for k, t in enumerate(txs) if t.role == "jammer"]
        )
        # Jammer dominates if the loudest jammer outpowers all non-jammer
        # interferers combined and is comparable to the best server.
        non_jam_interf_w = sum(
            rx_pow_row[k] for k, t in enumerate(txs)
            if t.role != "jammer" and k != best_j
        )
        jammer_dominates = bool(
            jammer_powers.size > 0 and float(jammer_powers.max()) > non_jam_interf_w
        )

        interf_dbm = 10 * np.log10(interf_w) + 30 if interf_w > 0 else float("-inf")

        per_rx_summary.append(
            RxSummary(
                rx_name=rx.name, rx_role=rx.role,
                best_server_tx=best_name, best_server_role=best_role,
                best_server_rx_power_dbm=best_dbm,
                interferer_power_dbm=interf_dbm,
                sinr_db=sinr_db,
                rogue_dominates=rogue_dominates,
                jammer_dominates=jammer_dominates,
            )
        )

    # Aggregates
    users = [s for s in per_rx_summary if s.rx_role == "user"]
    aggregate = {
        "num_users": len(users),
        "num_eavesdroppers": sum(1 for s in per_rx_summary if s.rx_role == "eavesdropper"),
        "users_with_rogue_dominant": sum(1 for s in users if s.rogue_dominates),
        "users_with_jammer_dominant": sum(1 for s in users if s.jammer_dominates),
        "median_user_sinr_db": float(np.median(
            [s.sinr_db for s in users if s.sinr_db is not None]
        )) if users else None,
        "noise_floor_dbm": n0_dbm,
    }

    return RunKpis(
        per_link_rows=per_link_rows,
        per_rx_summary=per_rx_summary,
        aggregate=aggregate,
    )
