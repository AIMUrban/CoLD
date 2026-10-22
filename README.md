# CoLD: Commuting OD Flow Prediction

This release package contains the final CoLD implementation and the commuting-flow datasets used in the experiments.

## Repository layout

```text
CoLD_release_package_20261022/
├── data/                         # Input data for the three cities
│   ├── NYCNTA/                   # 262 regions
│   ├── SF/                       # 242 regions
│   └── DC/                       # 206 regions
│       ├── attr.npy              # Regional attributes, shape [N, 48]
│       ├── dis.npy               # Pairwise geographic distances, shape [N, N]
│       └── od.npy                # OD flow matrix, shape [N, N]
├── splits/                       # Fixed 7:1:2 train/validation/test region splits
├── models/
│   ├── CoRD_release/             # Final model, data loading, training, and metrics
│   └── CoRD_research/            # Final command-line runner and analysis utilities
├── figures/                      # Empty folder for framework figures
├── outputs/                      # New checkpoints, logs, and predictions are written here
└── requirements.txt
```

`CoRD_release` is the final CoLD model implementation. It uses attribute-based attractive/emissive role encoders, geographic role-neighbor aggregation with a dynamic self-residual gate, separate inflow/outflow collaborative teachers, StopGrad contrastive distillation, role orthogonality regularization, and a row-column collaborative auxiliary predictor.

## Installation

Use Python 3.9+ and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA driver before running the commands below.

## Run the final model

Run commands from the package root. The example below trains on NYC with the released fixed split and writes all artifacts to `outputs/`.

```bash
python -m models.CoRD_research.run \
  --city-path data/NYCNTA \
  --split-path splits/NYCNTA_fixed_split.npz \
  --output-dir outputs/NYCNTA_seed0 \
  --seed 0 \
  --device cuda:0 \
  --neighbor-k 20
```

For SF and DC, replace `NYCNTA` with `SF` or `DC`, and use `--neighbor-k 30`:

```bash
python -m models.CoRD_research.run \
  --city-path data/SF \
  --split-path splits/SF_fixed_split.npz \
  --output-dir outputs/SF_seed0 \
  --seed 0 \
  --device cuda:0 \
  --neighbor-k 30
```

The runner selects the checkpoint with the lowest validation MSE. Each output directory contains `best.pt`, `summary.json`, `history.json`, `test_predictions.csv`, and `embeddings.npz` (unless disabled).

## Final default settings

The runner defaults to a 60-dimensional role representation, dynamic gate initialization of 0.2, Adam learning rate `1e-3`, weight decay `1e-5`, batch size 16, maximum 120 epochs, patience 40, `lambda_aux=1`, `lambda_con=2`, `lambda_orth=0.5`, temperature 0.5, and OD-similarity-weighted negative samples.

The fixed splits assign 70% of regions to training, 10% to validation, and 20% to testing. Collaborative OD profiles and auxiliary targets are built only from train-to-train OD flows; validation/test-region flows do not enter collaborative encoding during training.
