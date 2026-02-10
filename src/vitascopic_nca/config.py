from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class DefaultNCAConfig:
    message_channels: int = 12
    visual_channels: int = 3
    hidden_channels: int = 128
    fire_rate: float = 0.99
    alive_threshold: float = 0.1
    zero_initialization: bool = False
    mass_conserving: Literal["no", "normal", "cross_channel"] = "normal"
    padding_type: Literal["circular", "constant"] = "circular"
    beta: float = 1.0
    num_embs: int = 5
    msg_type: Literal["DNA", "random"] = "random"


@dataclass
class DefaultOptimizationConfig:
    loss_type: Literal["mse", "clf", "DNA"] = "DNA"
    lr: float = 0.0001
    batch_size: int = 24


@dataclass
class DefaultDecoderConfig:
    n_layers: int = 3
    hidden_dim: int = 128
    in_dim: int = 1
    pooling_fn: Callable = torch.amax


@dataclass
class DefaultTrainerConfig(
    DefaultNCAConfig,
    DefaultOptimizationConfig,
    DefaultDecoderConfig,
):
    H: int = 64
    W: int = 64
    device: str = "cuda"
    checkpoint_path: str = "./checkpoints"
