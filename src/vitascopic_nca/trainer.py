from dataclasses import dataclass
from typing import Literal

import torch

from vitascopic_nca.decoder import Decoder
from vitascopic_nca.nca import NeuralCA


@dataclass(frozen=True)
class DefaultNCAConfig:
    channels = 16
    hidden_channels = 128
    fire_rate = 0.9
    alive_threshold = 0.3
    zero_initialization = True
    padding_type: Literal["circular", "constant"] = "circular"


@dataclass(frozen=True)
class DefaultOptimizationConfig:
    loss_type: Literal["mse", "clf"] = "mse"
    lr = 1e-3
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


DEFAULT_TRAINER_CONFIG = DefaultTrainerConfig()


def sample_msg_generator(msg_size, device):
    def generator(batch_size):
        return torch.randn(batch_size, msg_size).to(device)

    return generator


class Trainer:
    def __init__(self, config):
        self.decoder = Decoder(
            in_dim=config.in_dim,
            latent_dim=config.channels,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            pooling_fn=config.pooling_fn,
        ).to(config.device)
        self.nca = NeuralCA(
            channels=config.channels,
            hidden_channels=config.hidden_channels,
            fire_rate=config.fire_rate,
            alive_threshold=config.alive_threshold,
            zero_initialization=config.zero_initialization,
            padding_type=config.padding_type,
        ).to(config.device)
        self.config = config
        if config.loss_type == "mse":
            self.msg_generator = sample_msg_generator(config.channels, config.device)
        else:
            raise NotImplementedError(f"Loss type {config.loss_type} not implemented.")
        self.history = []
        self.optim = torch.optim.Adam(
            list(self.nca.parameters()) + list(self.decoder.parameters()),
            lr=config.lr,
        )

    def _make_init_state(self, msg):
        state = torch.zeros(
            self.config.batch_size, self.nca.channels, self.config.H, self.config.W
        ).to(self.config.device)
        state[:, :, self.config.H // 2 + 1, self.config.W // 2 + 1] = msg
        return state

    def optim_step(self, steps):
        if type(steps) == tuple:
            l, r = steps
            steps = torch.randint(l, r, (1,)).item()

        msg = self.msg_generator(self.config.batch_size)
        initial_state = self._make_init_state(msg)
        out1 = self.nca(initial_state, steps=steps // 2)
        # apply noise here
        out2 = self.nca(out1[:, -1], steps=steps // 2)
        out_msg = self.decoder(out2[:, -1:, 0])

        if self.config.loss_type == "mse":
            loss = torch.mean((out_msg - msg) ** 2)
        else:
            raise NotImplementedError(
                f"Loss type {self.config.loss_type} not implemented."
            )

        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.history.append({"loss": loss.item()})

        info = {
            "loss": loss.item(),
            "steps": steps,
            "input_msg": msg.detach().cpu(),
            "output_msg": out_msg.detach().cpu(),
            "rollout": torch.cat([out1, out2], dim=1).detach().cpu(),
        }

        return info

    def sanity_check(self):
        info = self.optim_step(steps=8)
        loss1 = info["loss"]
        for _ in range(10):
            info = self.optim_step(steps=8)
        loss2 = info["loss"]
        print(f"Sanity check losses: {loss1} -> {loss2}")
        assert loss2 <= loss1, "Loss did not decrease in sanity check."


if __name__ == "__main__":
    trainer = Trainer(DEFAULT_TRAINER_CONFIG)
    trainer.sanity_check()
