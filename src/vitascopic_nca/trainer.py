import matplotlib.pyplot as plt
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
from vitascopic_nca.noise import Noiser
from vitascopic_nca.utils import image_row, impact_frames, sequence_batch_to_html_gifs


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

    def _make_init_state(self, msg):
        state = torch.zeros(
            self.config.batch_size, self.nca.channels, self.config.H, self.config.W
        ).to(self.config.device)

        if self.config.mass_conserving:
            state[
                :,
                0,
                self.config.H // 2 - 4 : self.config.H // 2 + 4,
                self.config.W // 2 - 4 : self.config.W // 2 + 4,
            ] = torch.tensor(1.0)

        state[:, :, self.config.H // 2, self.config.W // 2] = msg

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
        # gaussian_noise = torch.randn_like(final_frame) * 0.1
        # noised_final_frame = final_frame + gaussian_noise

        noised_final_frame = self.noiser(final_frame)
        out_msg = self.decoder(noised_final_frame)
        frames = [final_frame]
        noised_frames = [noised_final_frame]

        mass_threshold = 2000.0
        total_mass_loss = 0.5 * torch.relu(final_frame.sum() - mass_threshold)

        if self.config.loss_type == "mse":
            loss = torch.mean((out_msg - msg) ** 2)
        else:
            # loss = F.cross_entropy(out_msg, out)
            loss = F.mse_loss(out_msg, msg)

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

        def bar_plot(vec):
            fig, ax = plt.subplots(figsize=(2, 1))
            sns.barplot(x=list(range(len(vec))), y=vec)
            ax.set_xticks([])
            ax.set_ylabel("")
            plt.close()
            return pn.pane.Matplotlib(
                fig, format="svg", width=100, height=80, tight=True
            )

        return pn.Column(
            f"**Optimization Step (loss={info['loss']:.4f}, optim steps={self.learning_steps})**",
            pn.pane.Matplotlib(fig, format="svg", width=600, height=300, tight=True),
            f"""
            ```
            Rollout min max : {info['rollout'].min().item():.4f}, {info['rollout'].max().item():.4f}
            Rollout mean std: {info['rollout'].mean().item():.4f}, {info['rollout'].std().item():.4f}
            Total mass: {info['final_frame'].sum().item():.4f}
            ```
            """,
            pn.Row(*[bar_plot(info["input_msg"][i]) for i in range(to_show)]),
            pn.Row(*image_row(info["frames"], columns=to_show)),
            pn.Row(*image_row(info["noised_frames"], columns=to_show)),
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
        )

    def sanity_check(self):
        with torch.no_grad():
            info = self.optim_step(steps=10)
        loss = info["loss"]
        print(f"Sanity check loss: {loss}")

    def display_mass_sanity_check(self, info):
        rollout = info["rollout"]
        mass_channel = rollout[:, :, 0]
        mass_sums = mass_channel.view(
            mass_channel.shape[0], mass_channel.shape[1], -1
        ).sum(dim=-1)

        normalized_mass_sums = mass_sums / (mass_sums[:, :1] + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i in range(normalized_mass_sums.shape[0]):
            ax.plot(normalized_mass_sums[i].numpy(), alpha=0.6)
        ax.set_title("Mass channel sum over time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mass sum")
        plt.close()

        return pn.Column(
            "**Mass Conservation Check**",
            pn.pane.Matplotlib(fig, format="svg", width=600, height=300, tight=True),
        )

    def display_entropy(self, info):
        """Plot per-channel and global entropy over time for a single rollout sample."""
        rollout = info["rollout"]  # (B, T, C, H, W)
        sample = rollout[0]

        H_global = global_entropy_over_time(sample).cpu().numpy()
        H_channels = per_channel_entropy_over_time(sample).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        # for c in range(H_channels.shape[1]):
        for c in range(3):  # remember only first 3 channels are shown
            ax.plot(H_channels[:, c], alpha=0.6, label=f"Channel {c}")
        ax.plot(H_global, color="black", linewidth=2, label="Global entropy")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy over time (sample 0)")
        ax.legend(ncol=2, fontsize=8)
        plt.close()

        return pn.Column(
            "**Entropy Check**",
            pn.pane.Matplotlib(fig, format="svg", width=600, height=300, tight=True),
        )
