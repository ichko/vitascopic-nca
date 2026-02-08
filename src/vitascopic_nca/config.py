from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class DefaultNCAConfig:
    message_channels = 12
    visual_channels = 3
    hidden_channels = 128
    fire_rate = 0.9
    alive_threshold = 0.1
    zero_initialization = False
    mass_conserving: Literal["no", "normal", "cross_channel"] = "no"
    padding_type: Literal["circular", "constant"] = "circular"
    beta = 1.
    num_embs = 5
    msg_type: Literal["DNA", "random"] = "random"


@dataclass(frozen=True)
class DefaultOptimizationConfig:
    loss_type: Literal["mse", "clf"] = "mse"
    lr = 0.0001
    batch_size = 24


@dataclass(frozen=True)
class DefaultDecoderConfig:
    n_layers = 3
    hidden_dim = 32
    in_dim = 3
    pooling_fn = torch.amax


@dataclass(frozen=True)
class DefaultTrainerConfig(
    DefaultNCAConfig, DefaultOptimizationConfig, DefaultDecoderConfig
):
    H = 32
    W = 32
    device = "cpu"
    checkpoint_path = "./checkpoints"
