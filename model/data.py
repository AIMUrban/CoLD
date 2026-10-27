"""Region-level five-fold splits and strictly observed-region training inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    """One origin and all its eligible destinations (the batching unit)."""
    origin: int
    destinations: np.ndarray
    distances: np.ndarray
    flows: np.ndarray


@dataclass(frozen=True)
class CityData:
    attributes: np.ndarray
    distances: np.ndarray
    neighbors: np.ndarray
    inflow_profiles: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    train_samples: list[Sample]
    valid_samples: list[Sample]
    test_samples: list[Sample]


def validate_split(groups, count):
    groups = [np.asarray(group, dtype=np.int64) for group in groups]
    if len(groups) != 3 or any(group.ndim != 1 or not len(group) for group in groups):
        raise ValueError("A split requires nonempty 1-D train, validation and test sets")
    joined = np.concatenate(groups)
    if (len(joined) != count or len(np.unique(joined)) != count
            or joined.min() < 0 or joined.max() >= count):
        raise ValueError("Split must partition every region exactly once")
    return groups


def five_fold_splits(count, seed=0):
    """Each region is tested once; split the other 80% into train/val at 7:1."""
    if count < 10:
        raise ValueError("Five-fold train/validation/test splitting needs at least 10 regions")
    rng = np.random.default_rng(seed)
    test_folds = np.array_split(rng.permutation(count), 5)
    splits = []
    for test_idx in test_folds:
        remaining = rng.permutation(np.setdiff1d(np.arange(count), test_idx))
        valid_count = max(1, int(round(len(remaining) / 8)))
        valid_idx, train_idx = remaining[:valid_count], remaining[valid_count:]
        splits.append(tuple(validate_split(
            [np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)], count)))
    return splits


def _load_split(path, count, seed, fold):
    if not 0 <= fold < 5:
        raise ValueError("fold must be in 0..4")
    if path is None:
        return five_fold_splits(count, seed)[fold]
    with np.load(path, allow_pickle=False) as split:
        return validate_split([split[key] for key in ("train_idx", "valid_idx", "test_idx")], count)


def _samples(origins, destinations, od, distances):
    samples = []
    for origin in sorted(set(map(int, origins))):
        targets = np.asarray([j for j in destinations if j != origin], dtype=np.int64)
        if targets.size:
            samples.append(Sample(origin, targets, distances[origin, targets], od[origin, targets]))
    return samples


def _evaluation_samples(unseen, observed, od, distances):
    unseen = sorted(set(map(int, unseen)))
    observed = sorted(set(map(int, observed)))
    return (_samples(unseen, observed + unseen, od, distances)
            + _samples(observed, unseen, od, distances))


def observed_neighbors(distances, train_idx, neighbor_k):
    """All regions query only training regions; a training query excludes itself."""
    if not 0 < neighbor_k < len(train_idx):
        raise ValueError("neighbor_k must be positive and smaller than the training-region count")
    candidates = np.array(distances[:, train_idx], dtype=np.float64, copy=True)
    candidates[train_idx, np.arange(len(train_idx))] = np.inf
    order = np.argsort(candidates, axis=1, kind="stable")[:, :neighbor_k]
    return np.asarray(train_idx)[order]


def load_city(city_path, split_path=None, split_seed=0, neighbor_k=30, fold=0):
    city_path = Path(city_path)
    attributes = np.load(city_path / "attr.npy", allow_pickle=False)
    distances = np.load(city_path / "dis.npy", allow_pickle=False)
    od = np.load(city_path / "od.npy", allow_pickle=False)
    count = len(attributes)
    if attributes.ndim != 2 or distances.shape != (count, count) or od.shape != (count, count):
        raise ValueError("attr.npy, dis.npy and od.npy must describe the same regions")
    if not all(np.isfinite(a).all() for a in (attributes, distances, od)):
        raise ValueError("Input arrays must be finite")
    if (distances < 0).any() or (od < 0).any():
        raise ValueError("Distances and observed flows must be nonnegative")

    train_idx, valid_idx, test_idx = _load_split(split_path, count, split_seed, fold)
    neighbors = observed_neighbors(distances, train_idx, neighbor_k)
    train_sorted = sorted(map(int, train_idx))
    train_samples = _samples(train_sorted, train_sorted, od, distances)
    valid_samples = _evaluation_samples(valid_idx, train_idx, od, distances)
    # Validation endpoints are excluded from the test set, as in the manuscript.
    test_samples = _evaluation_samples(test_idx, train_idx, od, distances)

    train_od = od[np.ix_(train_idx, train_idx)].astype(np.float32, copy=True)
    np.fill_diagonal(train_od, 0.0)
    inflow_profiles = np.log1p(train_od.T).astype(np.float32)
    return CityData(attributes, distances, neighbors, inflow_profiles,
                    train_idx, valid_idx, test_idx, train_samples, valid_samples, test_samples)


def save_split(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = (data.train_idx, data.valid_idx, data.test_idx) if isinstance(data, CityData) else data
    np.savez(path, **dict(zip(("train_idx", "valid_idx", "test_idx"), groups)))

