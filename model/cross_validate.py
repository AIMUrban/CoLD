"""Run all five disjoint region test folds and report fold-wise means/stds."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .data import five_fold_splits, save_split
from .train import TrainConfig, fit


def aggregate_runs(runs):
    if len(runs) != 5 or sorted(run["fold"] for run in runs) != list(range(5)):
        raise ValueError("A five-fold summary requires exactly folds 0..4")
    arrays = []
    for run in runs:
        with np.load(run["split_path"], allow_pickle=False) as split:
            arrays.append(split["test_idx"])
    all_test = np.concatenate(arrays)
    n = sum(runs[0][key] for key in ("train_region_count", "valid_region_count", "test_region_count"))
    if len(all_test) != n or not np.array_equal(np.sort(all_test), np.arange(n)):
        raise ValueError("Test folds must be disjoint and cover every region once")
    def summarize(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "fold_values": list(map(float, values))}
    return {
        "city": runs[0]["city"], "fold_count": 5,
        "aggregation": "arithmetic mean of metrics independently computed per fold; sample std (ddof=1)",
        "metrics": {metric: summarize([r["test_metrics"][metric] for r in runs])
                    for metric in ("CPC", "MAE", "RMSE", "NRMSE")},
        "scenarios": {name: {metric: summarize([r["scenario_metrics"][name][metric] for r in runs])
                              for metric in ("CPC", "MAE", "RMSE", "NRMSE")}
                      for name in ("train_to_test", "test_to_train", "test_to_test")},
        "folds": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cities", nargs="+", choices=("NYCNTA", "SF", "DC"), default=["NYCNTA", "SF", "DC"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Training seed for fold f is seed+f")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negative-mode", choices=("od_weighted", "uniform"), default="od_weighted")
    parser.add_argument("--contrastive-view", choices=("separate", "concat"), default="separate")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.rglob("summary.json")):
        raise FileExistsError("Use a fresh output directory; existing results will not be overwritten")
    root.mkdir(parents=True, exist_ok=True)
    combined = {}
    for city in args.cities:
        city_path = (Path(args.data_root) / city).resolve()
        count = len(np.load(city_path / "attr.npy", allow_pickle=False))
        folds = five_fold_splits(count, args.split_seed)
        runs = []
        for fold, groups in enumerate(folds):
            split_path = root / city / "splits" / f"fold_{fold}.npz"
            save_split(split_path, groups)
            config = TrainConfig(
                city_path=str(city_path), output_dir=str(root / city / f"fold_{fold}"),
                split_path=str(split_path), split_seed=args.split_seed, fold=fold,
                seed=args.seed + fold, device=args.device, num_threads=args.num_threads,
                neighbor_k=20 if city == "NYCNTA" else 30,
                max_epoch=args.max_epoch, patience=args.patience, batch_size=args.batch_size,
                negative_mode=args.negative_mode, contrastive_view=args.contrastive_view,
            )
            print(f"\nCITY={city} FOLD={fold} CONFIG={json.dumps(asdict(config))}", flush=True)
            runs.append(fit(config))
        combined[city] = aggregate_runs(runs)
        (root / city / "cv_summary.json").write_text(json.dumps(combined[city], indent=2), encoding="utf-8")
        (root / "all_cities_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\n{city}: {json.dumps(combined[city]['metrics'])}", flush=True)


if __name__ == "__main__":
    main()
