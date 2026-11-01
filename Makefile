.PHONY: help install data train-m0 train-m1 train-m2 train-all simulate app test clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  install    : Install dependencies from requirements.txt"
	@echo "  data       : Fetch or synthesize and clean the carbon intensity dataset"
	@echo "  train-m0   : Train M0 (MSE point forecaster)"
	@echo "  train-m1   : Train M1 (Quantile pinball forecaster)"
	@echo "  train-m2   : Train M2 (Decision-focused regret forecaster)"
	@echo "  train-all  : Train all models across seeds"
	@echo "  simulate   : Run simulation benchmark over test horizons"
	@echo "  app        : Launch interactive Streamlit demo application"
	@echo "  test       : Run unit and integration tests with pytest"
	@echo "  clean      : Clean temporary python and checkpoint caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -c "from src.data import get_or_create_dataset; get_or_create_dataset()"

train-m0:
	$(PYTHON) -m src.train --config configs/train_m0.yaml --model-type m0

train-m1:
	$(PYTHON) -m src.train --config configs/train_m1.yaml --model-type m1

train-m2:
	$(PYTHON) -m src.train --config configs/train_m2.yaml --model-type m2

train-all: train-m0 train-m1 train-m2

simulate:
	$(PYTHON) -m src.simulate --config configs/sim.yaml

app:
	streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache checkpoints/*.tmp
