"""Train one CoLD region fold with the manuscript's losses and evaluation sets."""

import argparse
import copy
import csv
import hashlib
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import platform
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import load_city, save_split
from .metrics import regression_metrics
from .model import CoLD, ModelConfig


def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    city_path: str
    output_dir: str
    split_path: Optional[str] = None
    split_seed: int = 0
    fold: int = 0
    seed: int = 0
    device: str = "auto"
    num_threads: int = 4
    neighbor_k: int = 30
    max_epoch: int = 300
    patience: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_aux: float = 1.0
    lambda_con: float = 2.0
    lambda_orth: float = 0.5
    temperature: float = 0.5
    negative_mode: str = "od_weighted"  # od_weighted, uniform
    contrastive_view: str = "separate"  # separate, concat
    min_negative_weight: float = 0.0
    stopgrad_teacher: bool = True
    save_embeddings: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.negative_mode not in {"od_weighted", "uniform"}:
            raise ValueError(f"Unknown negative mode: {self.negative_mode}")
        if self.contrastive_view not in {"separate", "concat"}:
            raise ValueError(f"Unknown contrastive view: {self.contrastive_view}")
        if self.lambda_aux == 0 and self.lambda_con > 0 and self.stopgrad_teacher:
            raise ValueError("Without auxiliary loss, use joint distillation or disable contrastive loss")
        if self.max_epoch < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Epochs, batch size, and patience must be positive")
        if self.temperature <= 0 or self.lr <= 0 or self.num_threads < 1:
            raise ValueError("Temperature and learning rates must be positive")
        if min(self.lambda_aux, self.lambda_con, self.lambda_orth) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if not 0 <= self.min_negative_weight <= 1:
            raise ValueError("min_negative_weight must be in [0, 1]")
        if not 0 <= self.fold < 5:
            raise ValueError("fold must be in 0..4")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def negative_weights(profiles, minimum=0.0):
    normalized = F.normalize(profiles, dim=-1, eps=1e-12)
    similarity = (normalized @ normalized.T).clamp(-1, 1)
    weights = (1.0 - similarity).clamp(min=minimum, max=1.0)
    weights.fill_diagonal_(1.0)
    return weights.detach()


def contrastive_loss(student, teacher, temperature, weights=None):
    logits = F.normalize(student, dim=-1) @ F.normalize(teacher, dim=-1).T / temperature
    if weights is None:
        return F.cross_entropy(logits, torch.arange(len(student), device=student.device))
    # Exact zero negative weights contribute nothing; positive diagonal weights are 1.
    log_weights = torch.where(weights > 0, weights, torch.ones_like(weights)).log()
    log_weights = log_weights.masked_fill(weights == 0, -torch.inf)
    denominator = torch.logsumexp(logits + log_weights, dim=1)
    return (denominator - logits.diagonal()).mean()


def orthogonality_loss(attractive, emissive):
    cosine = (F.normalize(attractive, dim=-1) * F.normalize(emissive, dim=-1)).sum(dim=-1)
    return cosine.square().mean()


def evaluate(model, samples, features, neighbors):
    model.eval()
    predictions, targets, rows = [], [], []
    with torch.no_grad():
        region = model.region_encoder(features, neighbors)
        for sample in samples:
            destinations = torch.as_tensor(sample.destinations, device=features.device, dtype=torch.long)
            origin = region[sample.origin].unsqueeze(0).expand(len(destinations), -1)
            distance = torch.as_tensor(sample.distances, device=features.device, dtype=torch.float32).unsqueeze(-1)
            pred = model.predict_main(origin, region[destinations], distance)
            values = pred.cpu().numpy()
            predictions.append(values)
            targets.append(sample.flows)
            rows.extend((sample.origin, int(dst), float(value), float(flow)) for dst, value, flow in
                        zip(sample.destinations, values, sample.flows))
    metrics = regression_metrics(np.concatenate(predictions), np.concatenate(targets))
    return metrics, rows


def _save_embeddings(path, model, features, neighbors, profiles, train_idx):
    model.eval()
    with torch.no_grad():
        region, attractive, emissive = model.region_encoder.forward_with_roles(features, neighbors)
        pre_attractive, pre_emissive = model.region_encoder.get_pre_attention_role_vectors(features)
        inflow = model.inflow_encoder(profiles)
        outflow = model.outflow_encoder(profiles.T)
        arrays = {
            "region": region,
            "attractive": attractive,
            "emissive": emissive,
            "pre_attractive": pre_attractive,
            "pre_emissive": pre_emissive,
            "projected_attractive": model.attractive_projector(attractive),
            "projected_emissive": model.emissive_projector(emissive),
            "collaborative_inflow": inflow,
            "collaborative_outflow": outflow,
        }
        np.savez_compressed(path, train_idx=train_idx,
                            **{key: value.cpu().numpy() for key, value in arrays.items()})


def fit(config: TrainConfig):
    torch.set_num_threads(config.num_threads)
    set_seed(config.seed)
    data = load_city(config.city_path, config.split_path, config.split_seed, config.neighbor_k, config.fold)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "summary.json").exists():
        raise FileExistsError(f"Completed run already exists: {output}; use a new output directory")
    split_path = str(output / "split.npz")
    save_split(split_path, data)

    device = torch.device(default_device() if config.device == "auto" else config.device)
    features = torch.as_tensor(data.attributes, dtype=torch.float32, device=device)
    neighbors = torch.as_tensor(data.neighbors, dtype=torch.long, device=device)
    profiles = torch.as_tensor(data.inflow_profiles, dtype=torch.float32, device=device)
    train_idx = torch.as_tensor(data.train_idx, dtype=torch.long, device=device)
    lookup = torch.full((len(features),), -1, dtype=torch.long, device=device)
    lookup[train_idx] = torch.arange(len(train_idx), device=device)

    model = CoLD(features.size(1), len(train_idx), config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    metadata = {
        "config": asdict(config), "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "num_threads": torch.get_num_threads(),
        "batch_unit": "origin regions, each with all other training destinations",
        "source_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "data_sha256": {name: file_sha256(Path(config.city_path) / name)
                        for name in ("attr.npy", "dis.npy", "od.npy")},
        "split_sha256": file_sha256(split_path),
    }
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        (source_dir / source.name).write_bytes(source.read_bytes())
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    inflow_weights = outflow_weights = joint_weights = None
    if config.lambda_con > 0 and config.negative_mode == "od_weighted":
        inflow_weights = negative_weights(profiles, config.min_negative_weight)
        outflow_weights = negative_weights(profiles.T, config.min_negative_weight)
        if config.contrastive_view == "concat":
            joint_weights = negative_weights(torch.cat([profiles, profiles.T], dim=-1), config.min_negative_weight)

    best_mse = float("inf")
    best_state = None
    best_epoch = 0
    remaining = config.patience
    history = []
    for epoch in range(1, config.max_epoch + 1):
        model.train()
        loader = DataLoader(data.train_samples, batch_size=config.batch_size, shuffle=True, collate_fn=lambda batch: batch)
        losses = []
        component_losses = []
        positive_predictions = 0
        prediction_count = 0
        synchronize(device)
        train_started = time.perf_counter()
        for batch in loader:
            region, attractive, emissive = model.region_encoder.forward_with_roles(features, neighbors)
            if config.lambda_aux > 0 or config.lambda_con > 0:
                inflow = model.inflow_encoder(profiles)
                outflow = model.outflow_encoder(profiles.T)
            loss_con = torch.zeros((), device=device)
            if config.lambda_con > 0:
                student_in = model.attractive_projector(attractive[train_idx])
                student_out = model.emissive_projector(emissive[train_idx])
                teacher_in = inflow.detach() if config.stopgrad_teacher else inflow
                teacher_out = outflow.detach() if config.stopgrad_teacher else outflow
                if config.contrastive_view == "concat":
                    loss_con = contrastive_loss(
                        torch.cat([student_in, student_out], dim=-1),
                        torch.cat([teacher_in, teacher_out], dim=-1),
                        config.temperature, joint_weights,
                    )
                else:
                    loss_con = 0.5 * (
                        contrastive_loss(student_in, teacher_in, config.temperature, inflow_weights) +
                        contrastive_loss(student_out, teacher_out, config.temperature, outflow_weights)
                    )
            loss_orth = torch.zeros((), device=device)
            if config.lambda_orth > 0:
                pre_in, pre_out = model.region_encoder.get_pre_attention_role_vectors(features)
                loss_orth = orthogonality_loss(pre_in[train_idx], pre_out[train_idx])

            origins = torch.as_tensor([sample.origin for sample in batch], device=device, dtype=torch.long)
            destinations = torch.as_tensor(np.stack([sample.destinations for sample in batch]), device=device, dtype=torch.long)
            distances = torch.as_tensor(np.stack([sample.distances for sample in batch]), device=device, dtype=torch.float32).unsqueeze(-1)
            flows = torch.as_tensor(np.stack([sample.flows for sample in batch]), device=device, dtype=torch.float32).reshape(-1)
            origin_region = region[origins].unsqueeze(1).expand(-1, destinations.size(1), -1)
            prediction = model.predict_main(origin_region, region[destinations], distances)
            loss_main = F.mse_loss(prediction, flows)
            positive_predictions += int((prediction.detach() > 0).sum().item())
            prediction_count += prediction.numel()

            loss_aux = torch.zeros((), device=device)
            if config.lambda_aux > 0:
                origin_pos, destination_pos = lookup[origins], lookup[destinations]
                if (origin_pos < 0).any() or (destination_pos < 0).any():
                    raise ValueError("Training pairs must contain train regions only")
                out_origin = outflow[origin_pos].unsqueeze(1).expand(-1, destinations.size(1), -1)
                in_origin = inflow[origin_pos].unsqueeze(1).expand(-1, destinations.size(1), -1)
                prediction_aux = model.predict_aux(
                    out_origin, in_origin, outflow[destination_pos], inflow[destination_pos], distances,
                )
                loss_aux = F.mse_loss(prediction_aux, flows)
            loss = (loss_main + config.lambda_aux * loss_aux +
                    config.lambda_con * loss_con + config.lambda_orth * loss_orth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            component_losses.append([loss_main.item(), loss_aux.item(), loss_con.item(), loss_orth.item()])

        synchronize(device)
        train_seconds = time.perf_counter() - train_started

        valid_metrics, _ = evaluate(model, data.valid_samples, features, neighbors)
        if not np.isfinite(valid_metrics["MSE"]):
            raise FloatingPointError(f"Non-finite validation MSE at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "valid_mse": valid_metrics["MSE"], "valid_metrics": valid_metrics,
                        "train_seconds": train_seconds,
                        "train_positive_prediction_fraction": positive_predictions / prediction_count,
                        "loss_components": dict(zip(("main_mse", "collaborative_mse", "contrastive", "orthogonality"),
                                                    np.mean(component_losses, axis=0).tolist()))})
        print(f"epoch={epoch:03d} train={history[-1]['train_loss']:.4f} "
              f"valid_mse={valid_metrics['MSE']:.4f} valid_cpc={valid_metrics['CPC']:.4f} "
              f"train_seconds={train_seconds:.4f}", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if valid_metrics["MSE"] < best_mse:
            best_mse = valid_metrics["MSE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = config.patience
        else:
            remaining -= 1
            if remaining == 0:
                break

    model.load_state_dict(best_state)
    test_metrics, predictions = evaluate(model, data.test_samples, features, neighbors)
    train_set = set(map(int, data.train_idx))
    scenarios = {}
    for name, predicate in {
        "train_to_test": lambda i, j: i in train_set and j not in train_set,
        "test_to_train": lambda i, j: i not in train_set and j in train_set,
        "test_to_test": lambda i, j: i not in train_set and j not in train_set,
    }.items():
        selected = [row for row in predictions if predicate(row[0], row[1])]
        scenarios[name] = {"pairs": len(selected), **regression_metrics(
            np.asarray([row[2] for row in selected]), np.asarray([row[3] for row in selected]))}
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("origin_id", "dest_id", "predicted_flow", "true_flow"))
        writer.writerows(predictions)
    torch.save({"model": best_state, "config": asdict(config), "best_epoch": best_epoch}, output / "best.pt")
    if config.save_embeddings:
        _save_embeddings(output / "embeddings.npz", model, features, neighbors, profiles, data.train_idx)

    summary = {
        "city": Path(config.city_path).name,
        "seed": config.seed,
        "fold": config.fold,
        "split_path": split_path,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": remaining == 0,
        "best_valid_mse": best_mse,
        "train_pairs": sum(len(sample.destinations) for sample in data.train_samples),
        "valid_pairs": sum(len(sample.destinations) for sample in data.valid_samples),
        "test_pairs": len(predictions),
        "gate_statistics": model.region_encoder.gate_statistics(features, neighbors),
        "config": asdict(config),
        "test_metrics": test_metrics,
        "scenario_metrics": scenarios,
        "mean_train_seconds": float(np.mean([row["train_seconds"] for row in history])),
        "train_region_count": len(data.train_idx), "valid_region_count": len(data.valid_idx),
        "test_region_count": len(data.test_idx),
        "device": str(device),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-path", required=True)
    parser.add_argument("--split-path")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--neighbor-k", type=int, default=30)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negative-mode", choices=("od_weighted", "uniform"), default="od_weighted")
    parser.add_argument("--contrastive-view", choices=("separate", "concat"), default="separate")
    parser.add_argument("--lambda-con", type=float, default=2.0)
    parser.add_argument("--lambda-orth", type=float, default=0.5)
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()
    fit(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
