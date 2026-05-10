# Wireless-Sec-Sim — phase-by-phase reproducibility.
#
# Quick start:
#   make setup       # create venv + install deps
#   make repro       # run every phase end-to-end + validate
#   make validate    # cross-phase numerical checks against existing artefacts
#
# Override Mitsuba variant if needed:
#   make phase2 SIM_VARIANT=llvm_ad_mono_polarized

PYTHON      ?= .venv/bin/python
SIM_VARIANT ?= cuda_ad_mono_polarized
PYTEST_FLAGS = -p no:cacheprovider

ROOT := $(shell pwd)
DATA := $(ROOT)/data

.PHONY: help setup phase1 phase2 phase3 phase4 phase5 phase6 phase7 \
        baselines validate report test repro clean

help:
	@echo "Targets:"
	@echo "  setup     - Create venv and install requirements.txt"
	@echo "  phase1    - Munich coverage map (Sionna RT load + radio map)"
	@echo "  phase2    - Reference scenarios: baseline, jammed, rogue, FSPL"
	@echo "  phase3    - Reference sweeps: legit_bs / jammer / rogue position grids"
	@echo "  phase4    - Build dataset, train security classifier + coverage regressor"
	@echo "  phase5    - Fit embedder + ingest runs into vector store"
	@echo "  phase6    - Run optimization studies (perf + perf_sec)"
	@echo "  phase7    - Dashboard smoke tests"
	@echo "  baselines - TPE vs CMA-ES vs random comparison"
	@echo "  validate  - Cross-phase numerical checks → data/validation/report.json"
	@echo "  test      - Full pytest suite (isolated from system pytest plugins)"
	@echo "  repro     - End-to-end (phase1 → phase8)"
	@echo "  clean     - Remove generated artefacts under data/"

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

phase1:
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.load_scene --scene munich \
		--out $(DATA)/runs/munich_coverage.png

phase2:
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_scenario configs/munich_baseline.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_scenario configs/munich_jammed.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_scenario configs/munich_rogue.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_scenario configs/fspl_validation.yaml

phase3:
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_sweep configs/sweeps/legit_bs_position.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_sweep configs/sweeps/jammer_position.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_sweep configs/sweeps/rogue_position.yaml

phase4:
	$(PYTHON) -m scripts.build_dataset \
		$(DATA)/sweeps/legit_bs_position \
		$(DATA)/sweeps/jammer_position \
		$(DATA)/sweeps/rogue_position \
		--out $(DATA)/datasets/v1
	$(PYTHON) -m scripts.train_security_clf $(DATA)/datasets/v1 \
		--out $(DATA)/models/security_clf_v1
	$(PYTHON) -m scripts.train_coverage_predictor $(DATA)/sweeps/legit_bs_position \
		--out $(DATA)/models/coverage_predictor_v1

phase5:
	$(PYTHON) -m scripts.ingest_runs \
		--dataset $(DATA)/datasets/v1 \
		--embedder $(DATA)/embedders/v1 \
		--store $(DATA)/vector_store \
		$(DATA)/sweeps/legit_bs_position \
		$(DATA)/sweeps/jammer_position \
		$(DATA)/sweeps/rogue_position

phase6:
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.optimize_deployment configs/studies/munich_bs_perf.yaml
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.optimize_deployment configs/studies/munich_bs_perf_sec.yaml

phase7:
	PYTHONPATH= PYTHONNOUSERSITE=1 $(PYTHON) -m pytest tests/test_dashboard.py $(PYTEST_FLAGS)

baselines:
	SIM_VARIANT=$(SIM_VARIANT) $(PYTHON) -m scripts.run_baselines \
		configs/studies/munich_bs_perf_sec.yaml \
		--optimizers tpe cmaes random \
		--out $(DATA)/baselines/munich_bs_perf_sec

validate:
	$(PYTHON) -m scripts.validate_pipeline --out $(DATA)/validation/report.json

test:
	PYTHONPATH= PYTHONNOUSERSITE=1 $(PYTHON) -m pytest $(PYTEST_FLAGS)

repro: phase1 phase2 phase3 phase4 phase5 phase6 phase7 baselines validate
	@echo
	@echo "============================================================"
	@echo "  Reproduction complete. Validation report:"
	@echo "  $(DATA)/validation/report.json"
	@echo "============================================================"

clean:
	rm -rf $(DATA)/runs/* $(DATA)/sweeps/* $(DATA)/studies/* \
	       $(DATA)/datasets/* $(DATA)/models/* $(DATA)/embedders/* \
	       $(DATA)/vector_store/* $(DATA)/baselines/* $(DATA)/validation/*
