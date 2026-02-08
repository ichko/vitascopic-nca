import matplotlib.pyplot as plt
import numpy as np
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
    spatial_mass_entropy_over_time,
)
from vitascopic_nca.nca import NeuralCA
from vitascopic_nca.noise import Noiser
from vitascopic_nca.stimuli import Stimuli
from vitascopic_nca.utils import (
    image_row,
    impact_frames,
    plot_bars,
    sequence_batch_to_html_gifs,
    tensor_summary,
)


class SampleMsgGenerator(nn.Module):
    def __init__(self, msg_size, device, msg_type):
        super().__init__()
        self.msg_size = msg_size
        self.device = device
        self.msg_type = msg_type

        if msg_type == "DNA":
            assert (msg_size) % 4 == 0, "For DNA type, msg_size must bemultiple of 4"

    def forward(self, batch_size):
        if not self.msg_type == "DNA":
            x = torch.randn(batch_size, self.msg_size).to(self.device)
        else:
            # onehot encode 4xn bits
            x = torch.zeros(batch_size, self.msg_size).to(self.device)

            for i in range((x.shape[1])//4):
                indices = torch.randint(0, 4, (batch_size,)).to(self.device)
                x[:, i*4: (i+1)*4].scatter_(1, indices.unsqueeze(1), 1)
            

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
                config.message_channels if config.loss_type == "mse" else config.num_embs
            ),
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            pooling_fn=config.pooling_fn,
            padding_type=config.padding_type,
        ).to(config.device)
        self.nca = NeuralCA(
            message_channels=config.message_channels,
            hidden_channels=config.hidden_channels,
            fire_rate=config.fire_rate,
            alive_threshold=config.alive_threshold,
            zero_initialization=config.zero_initialization,
            padding_type=config.padding_type,
            mass_conserving=config.mass_conserving,
            beta=config.beta,
            visual_channels=config.visual_channels,
        ).to(config.device)

        self.config = config
        if config.loss_type == "mse":
            self.msg_generator = SampleMsgGenerator(config.message_channels, config.device, msg_type = config.msg_type)
        else:
            self.msg_generator = EmbeddingMsgGenerator(
                config.message_channels, num_embeddings=config.num_embs, device=config.device
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
            self.config.batch_size, self.nca.total_channels, self.config.H, self.config.W
        ).to(self.config.device)

        if self.config.mass_conserving == "normal":
            state[
                :,
                : self.config.visual_channels,
                self.config.H // 2 - 4 : self.config.H // 2 + 4,
                self.config.W // 2 - 4 : self.config.W // 2 + 4,
            ] = torch.tensor(2.0 , device=state.device)  # start with uniform mass distribution in visual channels
            # state[:, 0, :, :] = torch.tensor(1.0)  # start with uniform mass distribution
        elif self.config.mass_conserving == "cross_channel":
            raise NotImplementedError("Cross-channel mass conservation not implemented yet")
        else:
            state[
                :,
                : self.config.visual_channels,
                self.config.H // 2: self.config.H // 2,
                self.config.W // 2: self.config.W // 2,
            ] = torch.tensor(1.0 , device=state.device)  # to break up alivemasking

        state[:, self.config.visual_channels:, self.config.H // 2, self.config.W // 2] = msg

        return state

    def threshold_state(self, state):
        state = torch.where(state < 1.0, torch.zeros_like(state), state)
        return state

    def optim_step(self, steps):
        if type(steps) == tuple:
            l, r = steps
            steps = torch.randint(l, r, (1,)).item()

        msg, out = self.msg_generator(self.config.batch_size)

        initial_state = self._make_init_state(msg)

        # stimuli = Stimuli(initial_state=initial_state)

        # out1 = self.nca(initial_state, steps=5)
        # out1_usable = out1[:,-1]
        # initial_state = stimuli.add_stimuli(initial_state)

        out = self.nca(initial_state, steps=steps)

        final_frame = out[:, -1, :self.config.visual_channels]

        # noisesize = torch.sqrt(final_frame)
        # gaussian_noise = torch.randn_like(final_frame) * noisesize

        # noised_final_frame = final_frame + gaussian_noise * 0.8

        noised_final_frame = final_frame
        # noised_final_frame = self.noiser(noised_final_frame)
        # noised_final_frame = stimuli.add_stimuli_noise(noised_final_frame)

        # thresholded = (noised_final_frame >= 0.5).to(noised_final_frame.dtype)

        # if self.learning_steps % 300 == 0:
        #     if self.zeroing_thr < -2:
        #         self.zeroing_thr += 1
        #     print(self.learning_steps, self.zeroing_thr)

        # noised_final_frame[:, :, : self.zeroing_thr] = 0
        # print("NFF", noised_final_frame)
        
        # check if nans in noised_final_frame
        if torch.isnan(noised_final_frame).any():
            print("noised_final_frame has nans")
        
        # check final_frame stats
        if torch.isnan(final_frame).any():
            print("final_frame has nans")


        out_msg = self.decoder(noised_final_frame)
        # print("out_msg", out_msg)

        frames = [final_frame]
        noised_frames = [noised_final_frame]

        if self.config.loss_type == "mse":
            loss = F.mse_loss(out_msg, msg)
            # loss = F.binary_cross_entropy_with_logits(out_msg, msg)
        else:
            loss = F.cross_entropy(out_msg, out)

        grads = torch.tensor([0], dtype=torch.float32)
        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})
            # grads = [ This is makes things slow
            #     p.grad.detach().cpu() for p in self.parameters() if p.grad is not None
            # ]

        info = {
            "loss": loss.item(),
            "steps": steps,
            "input_msg": msg.detach().cpu(),
            "output_msg": out_msg.detach().cpu(),
            "final_frame": final_frame.detach().cpu(),
            "rollout": torch.cat([out], dim=1).detach().cpu(),
            "msg": msg,
            "frames": [f.detach().cpu() for f in frames],
            "noised_frames": [f.detach().cpu() for f in noised_frames],
            "grads": grads,
        }

        return info

    def display_optim_step(self, info):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.scatter(
            range(len(self.history)), [h["loss"] for h in self.history], s=1, alpha=0.9
        )
        ax.set_yscale("log")
        plt.close()

        to_show = 32
        steps = info["rollout"].shape[1] - 1

        # rollout: (B, T, C, H, W)

        rollout = info["rollout"][:to_show, :, :self.config.visual_channels]
        # rollout = impact_frames(rollout, ts=[0, steps], ns=[5, 20])
        # rollout = rollout[:, :, :self.config.visual_channels]
        rollout = rollout[:, :, :self.config.visual_channels]  # only show first 3 channels for visualization


        stats = f"""
            ```
            optim step: {self.learning_steps}
            frame  : {tensor_summary(info["rollout"])}
            weights: {tensor_summary(self.parameters())}
            grads  : {tensor_summary(info["grads"])}
            mass: {info['final_frame'].sum().item():.4f}
            ```
            """

        return pn.Row(
            pn.Column(
                pn.pane.Matplotlib(
                    fig, format="svg", width=500, height=250, tight=True
                ),
                pn.pane.HTML(
                    sequence_batch_to_html_gifs(
                        rollout,
                        columns=8,
                        width=100,
                        height=100,
                        fps=20,
                        return_html=True,
                    )
                ),
                # plot_bars(info["input_msg"][:to_show], info["output_msg"][:to_show]),
                image_row([f[:to_show] for f in info["frames"]], columns=to_show),
                image_row(
                    [f[:to_show] for f in info["noised_frames"]], columns=to_show
                ),
            ),
            pn.Column(
                stats,
                self.display_mass(info),
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

            ax.axhline(0, linewidth=0.5, color="tab:grey", linestyle="--")
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
        # ax.set_yscale("log")
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
        ax.set_yscale("log")
        plt.close()

        return pn.Column(
            "**Entropy Check**",
            pn.pane.Matplotlib(fig, format="svg", width=600, height=300, tight=True),
        )
