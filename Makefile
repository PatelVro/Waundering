.PHONY: help install gpu-check refresh views match-train ball-train sequence-train forecast live-track clean

PY      ?= python3
PIPE    := $(PY) -m cricket_pipeline.pipeline
DATASET ?= ipl_json
FMT     ?= T20,IT20

help:
	@echo "Cricket pipeline — make targets"
	@echo ""
	@echo "  make install            install deps + auto-detect GPU"
	@echo "  make install-gpu        install with LightGBM GPU build (best-effort)"
	@echo "  make gpu-check          run scripts/gpu_check.py"
	@echo "  make refresh            daily-refresh — re-pull, retrain match model"
	@echo "  make views              install/refresh analytical views"
	@echo "  make match-train        train the match-outcome model"
	@echo "  make ball-train         train the ball-outcome (LightGBM) model"
	@echo "  make sequence-train     train the sequence Transformer (uses GPU if avail)"
	@echo "  make forecast HOME=… AWAY=… VENUE=…   produce a full forecast"
	@echo "  make live-track         track today's live IPL match (auto-discover)"
	@echo "  make live-track MATCH_ID=123456   track a specific Cricbuzz match"
	@echo "  make clean              wipe the local DuckDB and cache"

install:
	./scripts/install.sh

install-gpu:
	./scripts/install.sh --lightgbm-gpu

gpu-check:
	$(PY) scripts/gpu_check.py

refresh:
	$(PIPE) daily-refresh --datasets $(DATASET) --fmt $(FMT)

views:
	$(PIPE) views

match-train:
	$(PIPE) match-train --fmt $(FMT) --device auto

ball-train:
	$(PIPE) model train --fmt T20 --device auto

sequence-train:
	$(PIPE) model train --type sequence --fmt T20 --epochs 8 --device auto

forecast:
	@if [ -z "$(HOME)" ] || [ -z "$(AWAY)" ] || [ -z "$(VENUE)" ]; then \
		echo "Usage: make forecast HOME='Lucknow Super Giants' AWAY='Kolkata Knight Riders' VENUE='Bharat …'"; \
		exit 2; \
	fi
	$(PIPE) match-forecast --home "$(HOME)" --away "$(AWAY)" --venue "$(VENUE)"

MATCH_ID ?=
INTERVAL ?= 60
N_SIM    ?= 1000

live-track:
	@if [ -n "$(MATCH_ID)" ]; then \
		$(PIPE) live-track --match-id "$(MATCH_ID)" \
			--home "$(HOME)" --away "$(AWAY)" \
			--interval $(INTERVAL) --n-sim $(N_SIM); \
	else \
		$(PIPE) live-track --auto \
			--home-hint "$(HOME)" --away-hint "$(AWAY)" \
			--interval $(INTERVAL) --n-sim $(N_SIM); \
	fi

clean:
	rm -f cricket_pipeline/data/cricket.duckdb
	rm -rf cricket_pipeline/data/cache/*
	rm -rf cricket_pipeline/data/models/*
	@echo "Wiped data/, cache/, models/"
