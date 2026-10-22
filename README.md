# CoLD: Commuting OD Flow Prediction

This release package contains the final CoLD implementation and the commuting-flow datasets used in the experiments.

## Framework

![CoLD framework](figures/framework.png)

## Repository layout

```text
CoLD/
├── data/                         # Input data for the three cities
│   ├── NYCNTA/                   # 262 regions
│   ├── SF/                       # 242 regions
│   └── DC/                       # 206 regions
│       ├── attr.npy              # Regional attributes, shape [N, 48]
│       ├── dis.npy               # Pairwise geographic distances, shape [N, N]
│       └── od.npy                # OD flow matrix, shape [N, N]
├── splits/                       # Fixed 7:1:2 train/validation/test region splits
├── models/
│   ├── data.py                   # Data loading and region splits
│   ├── metrics.py                # Evaluation metrics
│   ├── model.py                  # CoLD model architecture
│   └── train.py                  # Training and evaluation entry point
├── figures/                      # Framework figure in PNG format
│   └── framework.png
├── outputs/                      # New checkpoints, logs, and predictions are written here
├── README.md
└── requirements.txt
```

`models/model.py` contains the final CoLD model implementation. It uses attribute-based attractive/emissive role encoders, geographic role-neighbor aggregation with a dynamic self-residual gate, separate inflow/outflow collaborative teachers, StopGrad contrastive distillation, role orthogonality regularization, and a row-column collaborative auxiliary predictor.

## Installation

Use Python 3.9+ and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA driver before running the commands below.

## Run the final model

Run commands from the package root. The example below trains on NYC with the released fixed split and writes all artifacts to `outputs/`.

```bash
python -m models.train \
  --city-path data/NYCNTA \
  --split-path splits/NYCNTA_fixed_split.npz \
  --output-dir outputs/NYCNTA_seed0 \
  --seed 0 \
  --device cuda:0 \
  --neighbor-k 20
```

For SF and DC, replace `NYCNTA` with `SF` or `DC`, and use `--neighbor-k 30`:

```bash
python -m models.train \
  --city-path data/SF \
  --split-path splits/SF_fixed_split.npz \
  --output-dir outputs/SF_seed0 \
  --seed 0 \
  --device cuda:0 \
  --neighbor-k 30
```

The runner selects the checkpoint with the lowest validation MSE. Each output directory contains `best.pt`, `summary.json`, `history.json`, and `test_predictions.csv`.

