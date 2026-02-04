import matplotlib.pyplot as plt
import mediapy as media
import panel as pn
import torch

from vitascopic_nca.base_trainer import BaseTrainer
from vitascopic_nca.decoder import Decoder
from vitascopic_nca.nca import NeuralCA
from vitascopic_nca.utils import image_row, impact_frames, sequence_batch_to_html_gifs


class SampleMsgGenerator:
    def __init__(self, msg_size, device):
        self.msg_size = msg_size
        self.device = device

    def __call__(self, batch_size):
        return torch.randn(batch_size, self.msg_size).to(self.device)


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config.checkpoint_path)
        self.decoder = Decoder(
            in_dim=config.in_dim,
            latent_dim=config.channels,
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
        ).to(config.device)
        self.config = config
        if config.loss_type == "mse":
            self.msg_generator = SampleMsgGenerator(config.channels, config.device)
        else:
            raise NotImplementedError(f"Loss type {config.loss_type} not implemented.")
        self.history = []
        self.optim = torch.optim.Adam(
            list(self.nca.parameters()) + list(self.decoder.parameters()),
            lr=config.lr,
        )
        self.learning_steps = 0

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

        msg = self.msg_generator(self.config.batch_size)
        msg[:, 0] = 1  # Otherwise alive masking will not alow it to grow

        initial_state = self._make_init_state(msg)
        out1 = self.nca(initial_state, steps=steps)
        # out2 = self.nca(out1[:, -1], steps=steps // 2)
        final_frame = out1[:, -1:, 0]
        gaussian_noise = torch.randn_like(final_frame) * 0.1
        noised_final_frame = final_frame + gaussian_noise
        out_msg = self.decoder(noised_final_frame)
        frames = [final_frame]
        noised_frames = [noised_final_frame]

        mass_threshold = 2000.0
        total_mass_loss = 0.5 * torch.relu(final_frame.sum() - mass_threshold)

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

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})

        info = {
            "loss": loss.item(),
            "steps": steps,
            "input_msg": msg.detach().cpu(),
            "output_msg": out_msg.detach().cpu(),
            "final_frame": final_frame.detach().cpu(),
            "rollout": torch.cat([out1], dim=1).detach().cpu(),
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
            pn.Column(*image_row(info["frames"], columns=to_show)),
            pn.Column(*image_row(info["noised_frames"], columns=to_show)),
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
