import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from vitascopic_nca.base_trainer import BaseTrainer
from vitascopic_nca.decoder import Decoder
from vitascopic_nca.entropy_metrics import (
    global_entropy_over_time,
    per_channel_entropy_over_time,
)
from vitascopic_nca.nca import NeuralCA
from vitascopic_nca.utils import impact_frames, sequence_batch_to_html_gifs, image_row
from vitascopic_nca.entropy_metrics import (
    global_entropy_over_time,
    per_channel_entropy_over_time,
    spatial_mass_entropy_over_time,
)
from vitascopic_nca.stimuli import Stimuli
from vitascopic_nca.noise import Noiser
from vitascopic_nca.utils import (
    image_row,
    impact_frames,
    plot_bars,
    sequence_batch_to_html_gifs,
)


class SampleMsgGenerator(nn.Module):
    def __init__(self, msg_size, device):
        super().__init__()
        self.msg_size = msg_size
        self.device = device

    def forward(self, batch_size):
        x = torch.randn(batch_size, self.msg_size).to(self.device)
        return x, x


class EmbeddingMsgGenerator(nn.Module):
    def __init__(self, msg_size, num_embeddings, device):
        super().__init__()
        self.msg_size = msg_size
        self.num_embeddings = num_embeddings
        self.device = device
        self.embedding = nn.Embedding(num_embeddings, msg_size).to(device)

    def forward(self, batch_size):
        indices = torch.randint(0, self.num_embeddings, (batch_size,)).to(self.device)
        embs = self.embedding(indices)
        return embs, indices


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config.checkpoint_path)
        self.decoder = Decoder(
            in_dim=config.in_dim,
            latent_dim=(
                config.channels if config.loss_type == "mse" else config.num_embs
            ),
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            pooling_fn=config.pooling_fn,
            padding_type=config.padding_type,
        ).to(config.device)
        self.nca = NeuralCA(
            channels=config.channels,
            hidden_channels=config.hidden_channels,
            fire_rate=config.fire_rate,
            alive_threshold=config.alive_threshold,
            zero_initialization=config.zero_initialization,
            padding_type=config.padding_type,
            mass_conserving=config.mass_conserving,
            beta=config.beta,
        ).to(config.device)

        self.config = config
        if config.loss_type == "mse":
            self.msg_generator = SampleMsgGenerator(config.channels, config.device)
        else:
            self.msg_generator = EmbeddingMsgGenerator(
                config.channels, num_embeddings=config.num_embs, device=config.device
            )
        self.history = []
        self.optim = torch.optim.Adam(
            list(self.nca.parameters())
            + list(self.decoder.parameters())
            + list(self.msg_generator.parameters()),
            lr=config.lr,
        )
        self.learning_steps = 0
        self.noiser = Noiser()
        self.zeroing_thr = -17

    def _make_init_state(self, msg):
        state = torch.zeros(
            self.config.batch_size, self.nca.channels, self.config.H, self.config.W
        ).to(self.config.device)

        if self.config.mass_conserving == "normal":
            state[
                :,
                0,
                self.config.H // 2 - 4 : self.config.H // 2 + 4,
                self.config.W // 2 - 4 : self.config.W // 2 + 4,
            ] = torch.tensor(1.0)
            # state[:, 0, :, :] = torch.tensor(1.0)  # start with uniform mass distribution
        elif self.config.mass_conserving == "cross_channel":
            state[
                :,
                0,
                self.config.H // 2 - 4 : self.config.H // 2 + 4,
                self.config.W // 2 - 4 : self.config.W // 2 + 4,
            ] = torch.tensor(1.0)
            state[:, 1, :, :] = torch.tensor(6.0)  # start with uniform mass distribution


        state[:, :, self.config.H // 2, self.config.W // 2] = msg

        return state
    
    def threshold_state(self, state):
        state = torch.where(state < 1., torch.zeros_like(state), state)
        return state

    def optim_step(self, steps):
        if type(steps) == tuple:
            l, r = steps
            steps = torch.randint(l, r, (1,)).item()

        msg, out = self.msg_generator(self.config.batch_size)
        msg[:, 0] = 1  # Otherwise alive masking will not alow it to grow

        initial_state = self._make_init_state(msg)
        out1 = self.nca(initial_state, steps=steps)
        final_frame = out1[:, -1, :1]
        # noised_final_frame = self.noiser(final_frame)

        gaussian_noise = torch.randn_like(final_frame) * 0.5
        noised_final_frame = final_frame + gaussian_noise

        # thresholded = (noised_final_frame >= 0.5).to(noised_final_frame.dtype)

        # if self.learning_steps % 300 == 0:
        #     if self.zeroing_thr < -2:
        #         self.zeroing_thr += 1
        #     print(self.learning_steps, self.zeroing_thr)

        # noised_final_frame[:, :, : self.zeroing_thr] = 0
        out_msg = self.decoder(noised_final_frame)

        frames = [final_frame]
        noised_frames = [noised_final_frame]

        # mass_threshold = 2000.0
        # total_mass_loss = 0.5 * torch.relu(final_frame.sum() - mass_threshold)

        if self.config.loss_type == "mse":
            loss = torch.mean((out_msg - msg) ** 2)
        else:
            loss = F.cross_entropy(out_msg, out)

        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})

        info = {
            "loss": loss.item(),
            "steps": steps,
            "input_msg": msg.detach().cpu(),
            "output_msg": out_msg.detach().cpu(),
            "final_frame": final_frame.detach().cpu(),
            "rollout": torch.cat([out1], dim=1).detach().cpu(),
            "msg": msg,
            "frames": [f.detach().cpu() for f in frames],
            "noised_frames": [f.detach().cpu() for f in noised_frames],
        }

        return info

    def display_optim_step(self, info):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.scatter(
            range(len(self.history)), [h["loss"] for h in self.history], s=1, alpha=0.9
        )
        ax.set_yscale("log")
        plt.close()

        to_show = 8
        steps = info["rollout"].shape[1] - 1
        rollout = info["rollout"][:to_show, :, :1]
        rollout = impact_frames(rollout, ts=[0, steps], ns=[5, 20])
        rollout = rollout[:, :, 0]

        stats = f"""
            ```
            optim step: {self.learning_steps}
            min max : {info['rollout'].min().item():.4f}, {info['rollout'].max().item():.4f}
            mean std: {info['rollout'].mean().item():.4f}, {info['rollout'].std().item():.4f}
            mass: {info['final_frame'].sum().item():.4f}
            ```
            """

        return pn.Column(
            pn.Row(
                stats,
                pn.pane.Matplotlib(
                    fig, format="svg", width=500, height=250, tight=True
                ),
                self.display_mass(info),
            ),
            pn.pane.HTML(
                sequence_batch_to_html_gifs(
                    rollout,
                    columns=to_show,
                    width=100,
                    height=100,
                    fps=20,
                    return_html=True,
                )
            ),
            pn.Row(plot_bars(info["input_msg"][:to_show])),
            pn.Row(*image_row([f[:to_show] for f in info["frames"]], columns=to_show)),
            pn.Row(
                *image_row(
                    [f[:to_show] for f in info["noised_frames"]], columns=to_show
                )
            ),
        )

    def sanity_check(self):
        with torch.no_grad():
            info = self.optim_step(steps=10)
        loss = info["loss"]
        print(f"Sanity check loss: {loss}")

    def display_mass(self, info, normalize=False):
        rollout = info["rollout"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        for dim in reversed(range(rollout.shape[2])):
            mass_channel = rollout[:, :, dim]
            mass_sums = mass_channel.view(
                mass_channel.shape[0], mass_channel.shape[1], -1
            ).sum(dim=-1)

            if normalize:
                mass_sums = mass_sums / (mass_sums[:, :1] + 1e-8)

            for i in range(mass_sums.shape[0]):
                ax.plot(
                    mass_sums[i].numpy(),
                    alpha=1 if dim == 0 else 0.5,
                    linewidth=2 if dim == 0 else 0.1,
                    c="tab:orange" if dim == 0 else "tab:gray",
                )

        ax.set_title("Mass channel sum over time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mass sum")
        plt.close()

        return pn.pane.Matplotlib(fig, format="svg", width=500, height=250, tight=True)

    def display_entropy(self, info):
        """Plot per-channel and global entropy over time for a single rollout sample."""
        rollout = info["rollout"]  # (B, T, C, H, W)
        sample = rollout[0]

        H_global = global_entropy_over_time(sample).cpu().numpy()
        H_channels = per_channel_entropy_over_time(sample).cpu().numpy()
        H_mass = spatial_mass_entropy_over_time(sample.unsqueeze(0)).cpu().numpy()[0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for c in range(min(3, H_channels.shape[1])):  # show first 3 channels
            ax.plot(H_channels[:, c], alpha=0.5, label=f"Value ent ch {c}")
        ax.plot(H_mass, alpha=0.9, linestyle="--", label=f"Mass ent")

        ax.plot(H_global, color="black", linewidth=2, label="Global entropy (values)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy over time (sample 0)")
        ax.legend(ncol=2, fontsize=8)
        plt.close()

        return pn.Column(
            "**Entropy Check**",
            pn.pane.Matplotlib(fig, format="svg", width=600, height=300, tight=True),
        )
