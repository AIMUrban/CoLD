"""Attribute roles, collaborative roles, and OD predictors."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 60
    raw_hidden_dim: int = 120
    gate_init: float = 0.2
    gate_mode: str = "dynamic"  # dynamic, scalar, none
    use_roles: bool = True
    role_enhancement: bool = True
    use_raw_x: bool = True
    aux_mode: str = "both_roles"  # both_roles, directed
    use_projector: bool = True

    def __post_init__(self):
        if self.embed_dim < 1 or self.raw_hidden_dim < 1:
            raise ValueError("Embedding and raw hidden dimensions must be positive")
        if not self.use_roles and not self.use_raw_x:
            raise ValueError("At least one region representation branch must be enabled")
        if self.gate_mode not in {"dynamic", "scalar", "none"}:
            raise ValueError(f"Unknown gate mode: {self.gate_mode}")
        if self.aux_mode not in {"both_roles", "directed"}:
            raise ValueError(f"Unknown auxiliary mode: {self.aux_mode}")
        if not 0 < self.gate_init < 1:
            raise ValueError("gate_init must be between 0 and 1")


class SelfAttentionAggregator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, target, neighbors):
        query = self.q_proj(target).unsqueeze(1)
        keys = self.k_proj(neighbors)
        values = self.v_proj(neighbors)
        scores = (query * keys).sum(dim=-1) / math.sqrt(keys.size(-1))
        return (torch.softmax(scores, dim=-1).unsqueeze(-1) * values).sum(dim=1)


class AttrRoleEncoder(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.attr_norm = nn.LayerNorm(feature_dim)
        self.main = nn.Sequential(
            nn.Linear(feature_dim, embed_dim * 2), nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.skip = nn.Linear(feature_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        features = self.attr_norm(features)
        return torch.exp(self.log_scale) * self.out_norm(self.main(features) + self.skip(features))


class SpatialAttentionModel(nn.Module):
    def __init__(self, feature_dim, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.raw_attr_encoder = (
            nn.Sequential(
                nn.Linear(feature_dim, config.raw_hidden_dim), nn.GELU(),
                nn.Linear(config.raw_hidden_dim, dim),
            ) if config.use_raw_x else None
        )
        self.output_dim = dim * (2 * int(config.use_roles) + int(config.use_raw_x))
        self.in_encoder = AttrRoleEncoder(feature_dim, dim)
        self.out_encoder = AttrRoleEncoder(feature_dim, dim)
        self.in_agg = SelfAttentionAggregator(dim)
        self.out_agg = SelfAttentionAggregator(dim)
        if config.gate_mode != "none":
            initial_logit = math.log(config.gate_init / (1.0 - config.gate_init))
            self.in_residual_gate_logit = nn.Parameter(torch.tensor(initial_logit))
            self.out_residual_gate_logit = nn.Parameter(torch.tensor(initial_logit))
        if config.gate_mode == "dynamic":
            self.in_vector_gate_weight = nn.Parameter(torch.zeros(1, dim * 2))
            self.out_vector_gate_weight = nn.Parameter(torch.zeros(1, dim * 2))

    def get_pre_attention_role_vectors(self, features):
        return self.in_encoder(features), self.out_encoder(features)

    def _gates(self, attractive, emissive, neighbors_in, neighbors_out):
        if self.config.gate_mode == "none":
            return None, None
        if self.config.gate_mode == "scalar":
            return torch.sigmoid(self.in_residual_gate_logit), torch.sigmoid(self.out_residual_gate_logit)
        attractive_pair = torch.cat([attractive, neighbors_in], dim=-1)
        emissive_pair = torch.cat([emissive, neighbors_out], dim=-1)
        return (
            torch.sigmoid(F.linear(attractive_pair, self.in_vector_gate_weight) + self.in_residual_gate_logit),
            torch.sigmoid(F.linear(emissive_pair, self.out_vector_gate_weight) + self.out_residual_gate_logit),
        )

    def forward_with_roles(self, features, neighbors):
        attractive, emissive = self.get_pre_attention_role_vectors(features)
        if self.config.role_enhancement:
            role_in = self.in_agg(attractive, attractive[neighbors])
            role_out = self.out_agg(emissive, emissive[neighbors])
            gate_in, gate_out = self._gates(attractive, emissive, role_in, role_out)
            if gate_in is not None:
                role_in = role_in + gate_in * attractive
                role_out = role_out + gate_out * emissive
        else:
            role_in, role_out = attractive, emissive
        pieces = [role_in, role_out] if self.config.use_roles else []
        if self.raw_attr_encoder is not None:
            pieces.append(self.raw_attr_encoder(features))
        return torch.cat(pieces, dim=-1), role_in, role_out

    def forward(self, features, neighbors):
        return self.forward_with_roles(features, neighbors)[0]

    @torch.no_grad()
    def gate_statistics(self, features, neighbors):
        if self.config.gate_mode == "none" or not self.config.role_enhancement or not self.config.use_roles:
            return None
        attractive, emissive = self.get_pre_attention_role_vectors(features)
        neighbor_in = self.in_agg(attractive, attractive[neighbors])
        neighbor_out = self.out_agg(emissive, emissive[neighbors])
        gates = self._gates(attractive, emissive, neighbor_in, neighbor_out)
        result = {}
        for name, gate in zip(("attractive", "emissive"), gates):
            values = gate.expand(features.size(0), 1).float()
            result[name] = {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}
        return result


class MainPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair):
        return self.relation_mlp(pair).reshape(-1)


class LinearScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, scores):
        return torch.exp(self.log_scale) * scores


class ODJointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair):
        return self.mlp(pair).reshape(-1)


class ODFeatureMLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, profile):
        return self.mlp(profile)


class CoRD(nn.Module):
    def __init__(self, feature_dim, train_regions, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.region_encoder = SpatialAttentionModel(feature_dim, config)
        self.main_predictor = MainPredictor(self.region_encoder.output_dim * 2 + 1)
        self.flow_scale = LinearScaleLayer()
        aux_dim = dim * (4 if config.aux_mode == "both_roles" else 2) + 1
        self.aux_predictor = ODJointMLP(aux_dim, dim)
        self.inflow_encoder = ODFeatureMLP(train_regions, dim)
        self.outflow_encoder = ODFeatureMLP(train_regions, dim)
        self.attractive_projector = nn.Linear(dim, dim) if config.use_projector else nn.Identity()
        self.emissive_projector = nn.Linear(dim, dim) if config.use_projector else nn.Identity()

    def predict_main(self, origin, destination, distance):
        pair = torch.cat([origin, destination, distance], dim=-1)
        return self.flow_scale(self.main_predictor(pair))

    def predict_aux(self, out_origin, in_origin, out_destination, in_destination, distance):
        if self.config.aux_mode == "both_roles":
            pair = torch.cat([out_origin, in_origin, out_destination, in_destination, distance], dim=-1)
        else:
            pair = torch.cat([out_origin, in_destination, distance], dim=-1)
        return self.aux_predictor(pair)
