"""Region splits, OD profiles, and evaluation pairs."""

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np


@dataclass(frozen=True)
class Sample:
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


def _load_split(path, count, seed):
    if path is None:
        areas = [str(i) for i in range(count)]
        random.Random(seed).shuffle(areas)
        first, second = int(count * 0.7), int(count * 0.8)
        groups = [np.asarray([int(area) for area in group], dtype=np.int64)
                  for group in (areas[:first], areas[first:second], areas[second:])]
    else:
        with np.load(path) as split:
            groups = [np.asarray(split[key], dtype=np.int64) for key in
                      ("train_idx", "valid_idx", "test_idx")]
    joined = np.concatenate(groups)
    if len(joined) != count or len(np.unique(joined)) != count or joined.min() < 0 or joined.max() >= count:
        raise ValueError("Split must partition every region exactly once")
    return groups


def _samples(origins, destinations, od, distances):
    samples = []
    for origin in sorted(set(map(int, origins))):
        targets = np.asarray([j for j in destinations if j != origin], dtype=np.int64)
        if targets.size:
            samples.append(Sample(origin, targets, distances[origin, targets], od[origin, targets]))
    return samples


def _evaluation_samples(unseen, known, od, distances):
    unseen = sorted(set(map(int, unseen)))
    known = sorted(set(map(int, known)))
    forward_targets = list(dict.fromkeys(known + unseen))
    return _samples(unseen, forward_targets, od, distances) + _samples(known, unseen, od, distances)


def load_city(city_path, split_path=None, split_seed=0, neighbor_k=30):
    city_path = Path(city_path)
    attributes = np.load(city_path / "attr.npy")
    distances = np.load(city_path / "dis.npy")
    od = np.load(city_path / "od.npy")
    count = len(attributes)
    if distances.shape != (count, count) or od.shape != (count, count):
        raise ValueError("attr.npy, dis.npy, and od.npy must describe the same regions")
    if not 0 < neighbor_k < count:
        raise ValueError("neighbor_k must be between 1 and the number of other regions")

    train_idx, valid_idx, test_idx = _load_split(split_path, count, split_seed)
    dis_no_self = distances + np.eye(count) * np.max(distances) * 10
    neighbors = np.argsort(dis_no_self, axis=1)[:, :neighbor_k]

    train_sorted = sorted(map(int, train_idx))
    train_samples = _samples(train_sorted, train_sorted, od, distances)
    valid_samples = _evaluation_samples(valid_idx, train_idx, od, distances)
    known_for_test = list(dict.fromkeys([*map(int, train_idx), *map(int, valid_idx)]))
    test_samples = _evaluation_samples(test_idx, known_for_test, od, distances)

    train_od = od[np.ix_(train_idx, train_idx)].astype(np.float32)
    inflow = train_od.T
    with np.errstate(divide="ignore"):
        inflow_profiles = np.where(inflow >= 10, np.log(inflow), 0.0).astype(np.float32)
    return CityData(
        attributes=attributes, distances=distances, neighbors=neighbors,
        inflow_profiles=inflow_profiles, train_idx=train_idx,
        valid_idx=valid_idx, test_idx=test_idx, train_samples=train_samples,
        valid_samples=valid_samples, test_samples=test_samples,
    )


def save_split(path, data: CityData):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=data.train_idx, valid_idx=data.valid_idx, test_idx=data.test_idx)

