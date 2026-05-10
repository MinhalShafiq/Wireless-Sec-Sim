# Wireless-Sec-Sim

> A 3D AI-driven platform for **designing and securing** next-generation wireless networks.

`wireless-sec-sim` combines high-fidelity ray-traced RF propagation (Sionna RT 2.0), ML-based intrusion detection and coverage prediction, vector retrieval over a scenario library, and Bayesian optimization over deployment parameters. Every numerical claim below is backed by an artefact on disk and a check in `make validate`.

| | | |
|---|---|---|
| **FSPL agreement** | 0.006 dB max abs error vs. Friis | `configs/fspl_validation.yaml` |
| **Attack classifier** | F1 = 1.00 (24-scenario held-out) | `data/models/security_clf_v1` |
| **Coverage regressor** | RMSE = 0.83 dB / MAE = 0.26 dB | `data/models/coverage_predictor_v1` |
| **Retrieval precision@5** | 0.854 (hold-one-out, n = 93) | `data/vector_store` |
| **Optimization recovery** | +34 dB SINR vs. baseline under jammer | `data/studies/munich_bs_perf_sec` |
| **TPE vs random** | −2.57 vs −8.03 best objective (same budget) | `data/baselines/munich_bs_perf_sec` |
| **Test suite** | 38/38 passing | `make test` |

---

## Visuals

Same Munich scene, 3 different threats, full path-traced 3D renders:

| Baseline (no adversary) | Jammed (50 dBm jammer at centre) |
|---|---|
| ![baseline](docs/images/baseline_sinr.png) | ![jammed](docs/images/jammed_sinr.png) |

| Rogue gNB footprint | Jammer kill-zone heatmap (25-run sweep) |
|---|---|
| ![rogue](docs/images/rogue_footprint.png) | ![kill zone](docs/images/jammer_kill_zone.png) |

Optimizer head-to-head — same study, same seed, same trial budget:

![optimizer comparison](docs/images/optimizer_comparison.png)

---

## Quickstart

```bash
# 1. Set up the venv and dependencies
make setup

# 2. Run the entire eight-phase pipeline end-to-end (~5–10 min on RTX 2060)
make repro

# 3. Cross-phase numerical validation
make validate

# 4. Launch the interactive dashboard
streamlit run services/dashboard/Home.py
# or:    python -m scripts.dashboard
```

`make help` lists every target.

### One-shot examples (pieces of the pipeline)

```bash
# Phase 2 — single scenario with 3D views + KPIs
python -m scripts.run_scenario configs/munich_jammed.yaml

# Phase 3 — 25-point parameter sweep with attack-zone heatmap
python -m scripts.run_sweep configs/sweeps/jammer_position.yaml

# Phase 4 — train + predict
python -m scripts.train_security_clf data/datasets/v1 --out data/models/security_clf_v1
python -m scripts.predict_attack data/runs/munich_jammed --model data/models/security_clf_v1

# Phase 5 — find similar past scenarios
python -m scripts.find_similar data/runs/munich_jammed \
    --embedder data/embedders/v1 --store data/vector_store --k 5

# Phase 6 — optimize bs0 placement under a jammer
python -m scripts.optimize_deployment configs/studies/munich_bs_perf_sec.yaml --warm-start
```

---

## The eight-phase pipeline

| Phase | Purpose | Key files |
|---|---|---|
| **1. Environment** | 3D scene loading (Sionna RT built-in OSM scenes: Munich, Etoile, Florence, San Francisco) | `libs/scenes/`, `scripts/load_scene.py` |
| **2. Simulation engine** | Multi-TX/RX scenarios with adversaries, KPI extraction (SINR / best server / attack flags), photo-realistic 3D views | `libs/sim_engine/`, `libs/schemas.py`, `scripts/run_scenario.py` |
| **3. Sweeps** | Parameter grids over a base scenario, aggregate parquet, attack-zone heatmaps | `libs/sweep/`, `configs/sweeps/`, `scripts/run_sweep.py` |
| **4. ML / AI** | RandomForest attack classifier on role-free UE features + GradientBoosting path-gain regressor | `libs/ml/`, `scripts/train_security_clf.py`, `scripts/train_coverage_predictor.py`, `scripts/predict_attack.py` |
| **5. Retrieval** | Scenario embeddings → local Qdrant; nearest-neighbour decision support | `libs/retrieval/`, `scripts/ingest_runs.py`, `scripts/find_similar.py` |
| **6. Optimization** | Optuna TPE / CMA-ES / random over deployment params; joint perf + security objective; warm-start from Phase 5 | `libs/optim/`, `configs/studies/`, `scripts/optimize_deployment.py` |
| **7. Dashboard** | Streamlit multi-page app over every phase's artefacts | `services/dashboard/`, `scripts/dashboard.py` |
| **8. Validation & repro** | `Makefile` end-to-end, optimizer baseline harness, cross-phase numerical validation, technical report | `Makefile`, `scripts/run_baselines.py`, `scripts/validate_pipeline.py`, `docs/TECHNICAL_REPORT.md` |

Detailed per-phase methodology, results tables, and validation against analytical baselines: see [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

---

## Architecture

```
                                +---------------------------+
                                |  Streamlit dashboard      |  Phase 7
                                |  (services/dashboard)     |
                                +-------------+-------------+
                                              |
                                              v reads parquet/JSON/PNG
+--------+   +----------+   +----------+   +-----+--------+   +----------+
| Phase1 |   | Phase 2  |   | Phase 3  |   |  Phase 4     |   | Phase 6  |
| 3D env +-->| Sim eng  +-->| Sweep    +-->|  ML / AI     |<--+ Optimize |
| Sionna |   | KPIs+3D  |   | grids,   |   |  clf, reg    |   | Optuna   |
|  RT    |   | views    |   | parquet  |   |              |   | TPE/EA   |
+--------+   +----------+   +----------+   +------+-------+   +----+-----+
                                                  |                |
                                                  v                |
                                           +------+-------+        |
                                           |  Phase 5     |<-------+ warm start
                                           |  Vector store|
                                           |  Qdrant      |
                                           +--------------+
```

Each phase persists its artefacts under `data/<phase>/...`; downstream phases consume them as plain files. There is no orchestration service in the prototype — `make` is the dependency manager.

### Key technology choices

| Layer | Choice | Why |
|---|---|---|
| RF simulation | **Sionna RT 2.0.1** | GPU ray tracing via OptiX, built-in OSM scenes |
| Geometry backend | Mitsuba 3.8 + DrJit 1.3 | LLVM (CPU) ↔ CUDA (GPU OptiX) with one env-var flip |
| Schemas | Pydantic v2 | YAML round-trip, strict validation |
| Optimization | Optuna 4 (TPE / CMA-ES) | Sample-efficient, single API for multiple optimizers |
| Vector retrieval | qdrant-client (local-persistence mode) | API-identical to cloud Qdrant, no server |
| ML | scikit-learn 1.7 | Tabular RF / GB are strong baselines |
| Dashboard | Streamlit 1.57 | Reads parquet/JSON/PNG directly; multi-page out of the box |

---

## Repository layout

```
libs/
  scenes/         Sionna RT scene wrappers + Mitsuba variant selection
  sim_engine/     Scenario runner, KPI extractor, adversary injectors, 3D render, IO
  sweep/          Sweep schema, enumeration, runner, plotting             (Phase 3)
  ml/             Dataset, security classifier, coverage regressor         (Phase 4)
  retrieval/      Embedder, Qdrant store, decision support                 (Phase 5)
  optim/          Study schema, evaluator, Optuna runner, warm start       (Phase 6)
  schemas.py      Pydantic Scenario schema (YAML round-trip)

services/
  dashboard/      Streamlit multi-page app                                 (Phase 7)

scripts/          One CLI per pipeline action
configs/
  *.yaml          Single-Scenario configs                                  (Phase 2)
  sweeps/*.yaml   Sweep configs                                            (Phase 3)
  studies/*.yaml  Optimization study configs                               (Phase 6)

infra/docker/     Container image for the simulation worker
docs/             Technical report + visuals
tests/            pytest suite
data/             Generated artefacts (gitignored — reproduce via `make repro`)
```

---

## CPU vs GPU backend

Sionna RT supports both. Default in this repo is the **CPU LLVM backend** (works everywhere, no extra deps). For the GPU OptiX backend you need `libnvoptix.so.1` matched to your kernel-module version (`libnvidia-gl-<DRV>` on Ubuntu).

```bash
# CPU (default) — works everywhere
python -m scripts.run_scenario configs/munich_baseline.yaml

# GPU — requires OptiX runtime
SIM_VARIANT=cuda_ad_mono_polarized python -m scripts.run_scenario configs/munich_baseline.yaml
```

The `meta.json` in each run directory records which variant was used. **Numerical results agree across backends to within seed noise** (≪ 0.01 dB on FSPL, ≈ 0.001 dB on Munich SINR).

GPU prerequisites check:

```bash
cat /proc/driver/nvidia/version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
dpkg -l libnvidia-gl-* | tail -2
```

If the user-space lib version doesn't match the kernel module, downgrade/upgrade to align — consumer GPUs don't support forward compatibility.

---

## Reproducibility

```bash
make help        # list targets
make setup       # python -m venv .venv + pip install -r requirements.txt
make phase1      # Munich coverage map
make phase2      # Reference scenarios: baseline, jammed, rogue, FSPL
make phase3      # Reference sweeps: legit / jammer / rogue position
make phase4      # Build dataset, train classifier + coverage regressor
make phase5      # Fit embedder + ingest runs into vector store
make phase6      # Run optimization studies (perf + perf+security)
make phase7      # Dashboard smoke tests
make baselines   # TPE vs CMA-ES vs random head-to-head
make validate    # cross-phase numerical checks → data/validation/report.json
make test        # full pytest suite (isolated from system pytest plugins)
make repro       # phase1 → phase8 end-to-end
make clean       # remove generated artefacts under data/
```

`make validate` walks every phase's invariants (FSPL tolerance, sweep populations, classifier accuracy, coverage RMSE, retrieval precision@k, study improvements) and writes a structured report. Exits non-zero on any FAIL.

```text
[OK  ] phase1.scene_load        Munich extent 1475.5 × 1205.6 × 98.6 m, 11 objects
[OK  ] phase2.fspl_agreement    max abs error 0.006 dB (tol 0.5 dB)
[OK  ] phase3.sweeps_populated  legit 50 / jammer 25 / rogue 18 = 93 runs
[OK  ] phase4.ml_models         classifier acc/F1 = 1.000 / 1.000;  cov RMSE 0.83 dB
[OK  ] phase5.retrieval_quality precision@5 = 0.854 (n=93)
[OK  ] phase6.studies           perf-only +46.2 dB;  perf+sec +17.4 dB (under +50 dBm jammer)
6/6 passed in 4.46s
```

---

## Tests

The system Python path on Ubuntu desktops often includes ROS 2's pytest plugins, which interfere with discovery. Use the isolated invocation:

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 python -m pytest -p no:cacheprovider
# or simply:
make test
```

Targeted runs:

```bash
make test                                         # full suite, isolated
PYTHONPATH= PYTHONNOUSERSITE=1 pytest -k fspl     # just the Friis-agreement check
PYTHONPATH= PYTHONNOUSERSITE=1 pytest -m "not gpu"  # skip Sionna-dependent tests
```

The `gpu`-marked tests don't actually require a GPU — they need Sionna RT installed. The CPU LLVM backend is fine.

---

## Future work

The full list (with rationale) lives in [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) §6 and §8. The headline items:

1. **Custom OSM scenes** beyond the bundled Munich/Etoile/Florence/SF — `osmnx` is already installed; the import path → Sionna RT scene format is the missing glue. Unlocks cross-scene ML evaluation.
2. **Spoofed-pilot adversaries** — needs an NS-3 / 5G-LENA backend for link-layer modelling.
3. **Learned scenario embeddings** — autoencoder or contrastive encoder replacing the StandardScaler in Phase 5.
4. **CNN / GNN coverage models** — voxel-grid or building-graph inputs replacing the geometric tabular regressor.
5. **RL-based reconfiguration policies** — needs a temporal/mobility scenario model first.
6. **Cloud lift** — Argo Workflows over `scripts/`, hosted Qdrant via constructor swap, Postgres metadata for `data/sweeps` indexing, FastAPI gateway over the dashboard's read-only resources.

---

## License & references

Built on:

- [Sionna RT](https://github.com/NVlabs/sionna-rt) (NVIDIA) — differentiable ray-traced radio propagation
- [Optuna](https://optuna.org/) — hyperparameter and black-box optimization
- [Qdrant](https://qdrant.tech/) — vector database
- [scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/), [Pydantic](https://docs.pydantic.dev/), [Mitsuba 3](https://mitsuba.readthedocs.io/) / [DrJit](https://drjit.readthedocs.io/)

For methodology, full results, validation, and academic references: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
