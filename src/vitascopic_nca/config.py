from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class DefaultNCAConfig:
    channels = 16
    hidden_channels = 128
    fire_rate = 0.9
    alive_threshold = 0.1
    zero_initialization = True
    mass_conserving = True
    padding_type: Literal["circular", "constant"] = "circular"
    beta = 50
    num_embs = 2


@dataclass(frozen=True)
class DefaultOptimizationConfig:
    loss_type: Literal["mse", "clf"] = "clf"
    lr = 0.0001
    batch_size = 8


@dataclass(frozen=True)
class DefaultDecoderConfig:
    n_layers = 3
    hidden_dim = 32
    in_dim = 1
    pooling_fn = torch.amax


@dataclass(frozen=True)
class DefaultTrainerConfig(
    DefaultNCAConfig, DefaultOptimizationConfig, DefaultDecoderConfig
):
    H = 32
    W = 32
    device = "cuda"
    checkpoint_path = "./checkpoints"
