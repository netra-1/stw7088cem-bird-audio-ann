# Bird Audio Classification and Novelty Detection

This repository contains the implementation for the STW7088CEM Artificial Neural
Networks project on bird-call analysis. It addresses two related tasks:

1. Classifying recordings from 15 known bird species using EfficientNet-B0.
2. Detecting recordings from unknown species using a convolutional autoencoder.

Audio is converted into log-Mel spectrograms before modelling. Data is split by
recording group to reduce leakage, and predictions are evaluated at recording level.

## Setup

Python 3.11 and FFmpeg are required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
```

## Main commands

```bash
python -m bird_audio validate-configs
python -m bird_audio train-task1 --seed 37
python -m bird_audio train-task2 --seed 37
python -m bird_audio run-final-evaluation
```

Available commands and options can be viewed with:

```bash
python -m bird_audio --help
```

## Tests

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Repository structure

- `src/bird_audio/`: data preparation, models, training and evaluation
- `configs/`: experiment and model settings
- `scripts/`: supporting project scripts
- `tests/`: automated tests

## Data

The recordings were obtained from Xeno-canto. Audio files, processed datasets,
checkpoints and experiment outputs are not included in this repository.
