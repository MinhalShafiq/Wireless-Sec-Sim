# Wireless-Sec-Sim: A 3D AI-Driven Platform for Wireless Network Design and Security Evaluation

**Status.** Working prototype, single-machine reference implementation.
**Reference hardware.** Intel i7-10750H · 16 GB RAM · NVIDIA RTX 2060 (6 GB) · CUDA 13.0 · OptiX runtime 580.126.09 (matched to driver). The CPU (Mitsuba LLVM) backend reproduces every numerical result to within seed noise.

---

## 1. Abstract

This report documents the prototype implementation of a cloud-native 3D simulation platform for wireless network design and security evaluation. The platform combines high-fidelity ray-traced RF propagation (Sionna RT 2.0) with AI/ML for coverage prediction and intrusion detection, vector retrieval over a scenario library (Qdrant), and Bayesian / evolutionary optimization (Optuna) over deployment parameters. It is implemented as eight composable phases, each producing a self-contained artefact that downstream phases consume directly from disk.

Empirically:

- Path-gain measurements agree with the Friis free-space formula within **0.006 dB**.
- A binary attack/benign classifier on role-free UE features achieves **F1 = 1.0** on a 24-scenario held-out split (Munich corpus, 93 scenarios).
- A geometric path-gain regressor reaches **RMSE 0.83 dB / MAE 0.26 dB**.
- Vector retrieval over scenario embeddings hits **precision@1 = 0.957** and **precision@5 = 0.854** (hold-one-out).
- Bayesian optimization over a single base-station's placement, with a fixed centre-of-scene jammer in every trial, recovers median UE SINR from a baseline-collapsed **−16.9 dB** to **+17.4 dB** (a **34 dB** recovery) in **1.7 s** of wall-clock optimization on GPU.
- Head-to-head, TPE outperforms CMA-ES outperforms random search at the same trial budget.

All artefacts, configurations, seeds, and verification scripts ship with the repo; `make repro` reproduces every figure in this report.

---

## 2. System Architecture

### 2.1 Logical view

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
| (Sionna|   | (KPIs +  |   | (grids,  |   |  (clf, reg)  |   | (Optuna  |
|  RT)   |   |  3D view)|   |  parquet)|   |  ┌─────┐     |   |  TPE/EA) |
+--------+   +----------+   +----------+   |  └─────┘     |   +----+-----+
                                           +------+-------+        |
                                                  |                |
                                                  v                |
                                           +------+-------+        |
                                           |  Phase 5     |<-------+ warm start
                                           |  Vector store|
                                           |  (Qdrant)    |
                                           +--------------+
```

Each phase persists its artefacts under `data/<phase>/...`; downstream phases consume them as plain files. There is no orchestration service in the prototype — `make` is the dependency manager.

### 2.2 Repository layout (relevant subset)

```
libs/
  scenes/         Sionna RT scene wrappers + Mitsuba variant selection
  sim_engine/     Scenario runner, KPI extractor, adversary injectors, 3D render, IO
  sweep/          Sweep schema, enumeration, runner, plotting (Phase 3)
  ml/             Dataset, security classifier, coverage regressor (Phase 4)
  retrieval/      Embedder, Qdrant store, decision support (Phase 5)
  optim/          Study schema, evaluator, Optuna runner, warm start (Phase 6)
  schemas.py      Pydantic v2 Scenario schema (YAML round-trip)
services/
  dashboard/      Streamlit multi-page app (Phase 7)
scripts/          One CLI per pipeline action (run_scenario, run_sweep, …)
configs/
  *.yaml          Single-Scenario configs
  sweeps/*.yaml   Sweep configs
  studies/*.yaml  Optimization study configs
infra/docker/     Container image
docs/             Design proposal, this report
```

### 2.3 Key technology choices

| Layer | Choice | Why |
|---|---|---|
| RF simulation | **Sionna RT 2.0.1** | GPU ray tracing via OptiX, differentiable, built-in Munich/Etoile/Florence/SF scenes |
| Geometry backend | Mitsuba 3.8 + DrJit 1.3 | LLVM (CPU) ↔ CUDA (GPU OptiX) with one env-var flip |
| Schemas | Pydantic v2 | YAML round-trip, strict validation |
| Optimization | **Optuna 4** | TPE + CMA-ES + random in one API; numpy-2 friendly |
| Vector retrieval | **qdrant-client** local-persistence mode | API-identical to cloud Qdrant, no server |
| ML | scikit-learn 1.7 | Tabular RandomForest / GradientBoosting are strong baselines |
| Dashboard | Streamlit 1.57 | Reads our parquet/JSON/PNG directly; multi-page nav out of the box |

---

## 3. Methodology by phase

### 3.1 Phase 1 — 3D environment

We adopt Sionna RT's bundled OSM-derived urban scenes (Munich, Etoile, Florence, San Francisco) plus several primitive geometries. Munich is the canonical reference: ~1.5 km × 1.2 km × 100 m, 11 mesh objects, Frauenkirche centred at the origin. Switching environments is a single config field; an OSM-import path for custom scenes is wired through `osmnx` + Blender (Phase 1.5, future).

### 3.2 Phase 2 — simulation engine

**Schema.** A `Scenario` (Pydantic) is the platform's unit of work: scene + frequency band + transmitters + receivers + radio-map config + propagation parameters. Transmitters carry a role tag (`legitimate` / `rogue` / `jammer`); receivers carry `user` / `eavesdropper`.

**Path-level KPIs.** For each candidate scenario we run Sionna RT's `PathSolver`, sum |coef|² over valid paths to get the per-(RX, TX) linear path gain, convert to dBm with the configured TX power, and derive: best legitimate server, second-best, serving margin, total interference, SINR (with thermal noise + noise figure), rogue-dominance flag, jammer-dominance flag.

**Area KPIs.** Optionally we also run `RadioMapSolver` to produce a 2-D path-gain / RSS / SINR map at a given height + cell size.

**3D views.** Per run we render up to 5 photo-realistic images via Sionna RT's path tracer: `view_overview` (geometry + markers), `view_coverage` (radio map projected onto the ground for the primary legitimate TX), `view_sinr` (combined SINR; attack zones show as dark patches), `view_paths` (ray paths drawn TX→RX), `view_<role>_<name>` (per-adversary footprint, one per rogue/jammer).

**Validation.** A dedicated scenario in `floor_wall` with depth-0 LOS-only ray tracing compares simulated path gain against the analytical Friis formula. The configured tolerance is 0.5 dB; observed maximum absolute error is **0.006 dB**.

**Adversaries.**
- Jammer — extra TX with arbitrary power, role-tagged. Default 50 dBm.
- Rogue gNB — extra TX mimicking the legitimate signature (matches base BS power).
- Eavesdropper — passive RX role. Used in Phase 6+ for secrecy capacity.
- *Spoofed pilots are deferred — they need link-layer / pilot-aware modeling beyond ray tracing alone (NS-3 backend, future).*

### 3.3 Phase 3 — parameter sweeps

The local sweep runner enumerates `grid × seeds` combos, applies a template (`none` / `jammer` / `rogue` / `legit_tx`) to a base scenario, executes the sim engine, and persists per-run artefacts under `runs/<run_id>/`. A sweep-level `summary.parquet` plus auto-generated heatmaps (`plot_xy_*.png`) and an SINR distribution histogram capture cross-run patterns.

Three reference sweeps:

| Sweep | Template | Cardinality | Purpose |
|---|---|---|---|
| `legit_bs_position` | `legit_tx` (no adversary) | 50 | Diverse benign deployments — feeds Phase 4's negative class |
| `jammer_position` | `jammer` | 25 | Attack-zone exploration over jammer (x, y) |
| `rogue_position` | `rogue` | 18 | Best-server-flip surface over rogue (x, y) × power |

Throughput: ~1 s per run on the RTX 2060 (typical Munich path solver, depth 3, 500k samples).

### 3.4 Phase 4 — ML / AI

**Dataset.** Per scenario we extract role-free per-UE features (best signal, second-best, serving margin, sum-of-RX-power, n strong TXs, max/min/spread of path gain, distances to each TX) and aggregate (mean / std / min / max / median) into a 63-dim feature vector. Label = 1 iff any adversary is present in the scenario.

**Security classifier.** RandomForest pipeline (median imputer → standard scaler → balanced-class RF). Stratified scenario-level train/test split (75/25). Reports accuracy, F1, ROC AUC, confusion matrix, feature importances.

**Coverage / path-gain regressor.** Geometric features only: (`tx_xyz`, `rx_xyz`, deltas, distance, log-distance, height diff, carrier). GradientBoosting baseline. Trained on the benign sweep's per-link rows.

**Inference.** `predict_attack <run_dir>` returns `{predicted_label, p_attack}`.

**Deferred.** CNN over 2-D path-loss, GNN over building/AP graph, ART adversarial-robustness probing, MLflow / Hydra / FastAPI inference service.

### 3.5 Phase 5 — vector retrieval

**Embedder.** `StandardScaler` over the 63 Phase-4 features, fit on the training dataset, persisted to `data/embedders/v1/embedder.joblib`. The proposal's contrastive / autoencoder embedder is the natural successor — only `ScenarioEmbedder.embed_run` needs to change; the store API and decision-support code are agnostic.

**Store.** `qdrant-client` in **local persistence mode** (no server). One collection (`scenarios`), cosine distance, deterministic UUID5 point IDs (idempotent re-ingestion), payload includes label, scene, has_jammer, has_rogue, n_tx, carrier, key KPIs, source root.

**Decision support.** `summarize_similar(run, k, filters)` returns matches + a one-line outcome rollup (`share_attack`, neighbour median SINR, any-jammer / any-rogue flags). Filters use Qdrant's payload constraints; switching to a hosted Qdrant is a one-line constructor change.

### 3.6 Phase 6 — optimization & security

**Search space.** Per-TX continuous ranges over (`position_x`, `position_y`, `position_z`, `power_dbm`). YAML-declared, `null` ranges fix that knob.

**Threat injection.** Optional `ThreatCfg` adds the same fixed jammer / rogue gNB to every trial's scenario, so the optimizer faces a consistent adversary while it explores deployment parameters.

**Objective.** `direction × Σ(weight × metric)`, scalarised across whatever subset of the aggregate KPIs you specify (median UE SINR, jammer-dominated UE count, etc.). Multi-objective (Pareto) is one Optuna constructor flag away.

**Optimizers.** TPE (default), CMA-ES, random — all via Optuna. RL deferred until a temporal scenario model is in scope.

**Speed trick.** `StudyEvaluator` keeps the loaded scene + RX layout + TX/RX arrays alive across trials. Each trial just removes/re-adds transmitters and re-runs the path solver; no scene reload, no per-trial disk IO. Result: **~0.07 s/trial** on GPU vs. ~1 s/trial via the full `run_scenario` pipeline.

**Warm start.** Optional `enqueue_trial`-based warm start that pulls similar past scenarios from the Phase-5 vector store and seeds them as initial trials.

### 3.7 Phase 7 — dashboard

A four-page Streamlit app: corpus inventory, per-run browser (3D view gallery + KPI strip + per-link parquet), sweep results (heatmaps + interactive scatter), retrieval explorer (query → side-by-side 3D views of query vs. top match), study results (convergence plot + best params + interactive trial scatter). The dashboard reads the same files the CLIs write — no backend service.

### 3.8 Phase 8 — validation & reproducibility

- **Reproducibility.** `make repro` runs every phase end-to-end and emits a validation report.
- **Cross-phase validation.** `scripts/validate_pipeline.py` checks the on-disk artefacts of every phase against numerical thresholds (FSPL tolerance, sweep populations, classifier accuracy, coverage RMSE, retrieval precision@k, study improvements). Returns non-zero exit code on any failure.
- **Optimizer baselines.** `scripts/run_baselines.py` runs the same `munich_bs_perf_sec` study under TPE / CMA-ES / random with identical seeds and writes a head-to-head plot.

---

## 4. Results

All numbers below are **reproduced live** every time `make validate` runs.

### 4.1 Cross-phase validation report

| Phase | Check | Value | Threshold | Status |
|---|---|---|---|---|
| 1 | Munich bbox extent (x, y, z) m | 1475.5 × 1205.6 × 98.6 | > 500 × 500 × 50 | ✅ |
| 2 | FSPL agreement (max abs error) | **0.006 dB** | < 0.5 dB | ✅ |
| 3 | Reference sweeps populated | 50 + 25 + 18 = 93 runs | exact | ✅ |
| 4 | Classifier accuracy / F1 | 1.000 / 1.000 (n_test = 24) | ≥ 0.85 | ✅ |
| 4 | Coverage regressor RMSE | **0.83 dB** (n_test = 21) | < 5 dB | ✅ |
| 5 | Retrieval precision@5 (hold-one-out, n = 93) | **0.854** | ≥ 0.7 | ✅ |
| 6 | Best perf-only median SINR | **+46.2 dB** | ≥ 30 dB | ✅ |
| 6 | Best perf+sec median SINR (under 50 dBm jammer) | **+17.4 dB** | > 0 dB | ✅ |

`6 / 6 checks passed in 4.46 s` (verbatim from `scripts/validate_pipeline.py`).

### 4.2 Attack semantics flow through the simulator

Same Munich scene, same UEs, varying threat:

| Scenario | Median UE SINR | Attack flag |
|---|---|---|
| `munich_baseline` (3 BSs, no adversary) | **+41 dB** | — |
| `munich_jammed` (+ 50 dBm jammer at scene centre) | **−17 dB** (Δ = 58 dB) | 1 / 4 UEs jammer-dominated |
| `munich_rogue` (rogue gNB mimicking legitimate signature) | **−1 dB** | 2 / 3 UEs see rogue as strongest signal |

### 4.3 Retrieval quality (hold-one-out)

| k | precision@k mean | precision@k pooled |
|---|---|---|
| 1 | **0.957** | 0.957 |
| 3 | 0.882 | 0.882 |
| 5 | 0.854 | 0.854 |
| 10 | 0.837 | 0.837 |

### 4.4 Optimizer head-to-head (`munich_bs_perf_sec`, 25 trials)

Same study config (base scenario, search space, threat, objective, seed 42). Direction: maximise; objective = median_user_sinr_db − 20 × users_with_jammer_dominant.

| Optimizer | Best objective | Best median SINR | Wall time |
|---|---|---|---|
| **TPE** | **−2.57** | **+17.4 dB** | 1.6 s |
| CMA-ES | −4.19 | +15.8 dB | 1.7 s |
| Random | −8.03 | +12.0 dB | 1.6 s |

TPE finds a 5.5-objective-units better solution than random search at identical trial budget. CMA-ES converges faster (good objective by trial 8) but TPE wins eventually via better exploration around trial 19.

### 4.5 Optimization recovers from threat

Same threat, same scene. The baseline placement (3 BSs at the configured Munich positions) under the +50 dBm jammer collapses to median UE SINR −16.9 dB. Re-optimising bs0's position with TPE for 25 trials lifts it to **+17.4 dB** — a **34 dB recovery** in 1.7 s of wall clock.

---

## 5. Validation against analytical baselines

- **Free-space path loss.** Friis formula, max abs error 0.006 dB across 3 distance points (50 m, 100 m, 200 m) at 3.5 GHz with depth-0 LOS-only tracing in the `floor_wall` scene. Both LLVM and CUDA backends.
- **Cross-backend numerical agreement.** CPU LLVM vs GPU CUDA backends produce identical aggregate KPIs to within seed-noise: FSPL test ≪ 0.01 dB; Munich median SINR ≈ 0.001 dB.

---

## 6. Limitations

- **Single scene corpus.** All numerical results are on Munich. Cross-scene generalisation (Etoile, San Francisco, custom OSM extracts) is future work; the design supports it directly.
- **Static UE population.** UEs are placed and held; no mobility, no time-series. RL-based reconfiguration (proposal Phase 6) waits on this.
- **Tabular ML baselines only.** CNN over 2-D path-loss / GNN over building-AP graph / ART adversarial robustness probing — proposal targets, deferred. Tabular baselines validate the data flow and provide a strong regression baseline.
- **No standards-aligned MAC/RLC.** Spoofed pilots, beam-management attacks, and similar link-layer adversaries need NS-3 / 5G-LENA bridging.
- **Cloud plumbing.** Argo / Kubernetes / FastAPI / MLflow / Postgres are deferred consistently across phases. The Streamlit dashboard reads files directly. The `infra/docker/Dockerfile` packages the simulation worker; the lift to cloud is straightforward but not exercised here.

---

## 7. Reproducibility

```bash
make setup          # venv + pip install -r requirements.txt
make repro          # phase1 → phase8, end to end
make validate       # cross-phase numerical checks, exits non-zero on any FAIL
```

Every result above is a function of: the pinned dependency versions in `requirements.txt`, the seeds in the config YAMLs (Phase 2/3 default 42, Phase 6 default 42), and the choice of Mitsuba variant (`SIM_VARIANT=cuda_ad_mono_polarized` for the numbers in this report; `SIM_VARIANT=llvm_ad_mono_polarized` for CPU). The Docker image (`infra/docker/Dockerfile`) freezes the userspace environment.

---

## 8. Future work (calls forward to the design proposal)

1. **Custom OSM scenes (Phase 1.5).** `osmnx` is already installed; the import path → Sionna RT scene format is the missing glue. Unlocks cross-scene ML evaluation.
2. **Spoofed pilots + link-layer adversaries.** NS-3 / 5G-LENA backend.
3. **Learned scenario embeddings.** Contrastive or autoencoder encoder replacing the StandardScaler. The retrieval API is already abstracted.
4. **CNN / GNN coverage models.** Inputs: voxelized geometry, building graph. Drop-in via `libs.ml` interfaces.
5. **Multi-objective Pareto search.** Optuna constructor flag.
6. **RL reconfiguration policies.** Needs a temporal/mobility scenario model first.
7. **Cloud lift.** Argo workflows over the existing `scripts/`, Postgres metadata for `data/sweeps` indexing, FastAPI gateway over the dashboard's read-only resources, hosted Qdrant via constructor swap.

---

## 9. References

- Hoydis, J., Cammerer, S., et al. (2023). *Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling*. arXiv:2303.11103.
- NVIDIA. (2024–2026). *Sionna 2.0 documentation* — <https://nvlabs.github.io/sionna/>.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019.
- Qdrant. (2024). *Qdrant — high-performance vector search*. <https://qdrant.tech/>.
- ITU-R P.2040-3 (2023). Effects of building materials and structures on radiowave propagation above about 100 MHz.
- 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*.

Phase-internal references (in-repo): `design_proposal.md`, `README.md`, the per-phase YAML configs, and the per-test docstrings.
