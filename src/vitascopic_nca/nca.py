import torch
import torch.nn as nn
import torch.nn.functional as F

from vitascopic_nca.mass_conservation import (
    cross_channel_mass_conserving_update,
    mass_conserving_update,
)


class NeuralCA(nn.Module):
    def alive(self, x, alive_threshold):
        return (
            F.max_pool2d(x[:, :self.visual_channels, :, :], kernel_size=3, stride=1, padding=0).amax(dim=1, keepdim=True)
            > alive_threshold 
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        message_channels,
        hidden_channels,
        fire_rate,
        alive_threshold,
        zero_initialization,
        mass_conserving,
        beta,
        visual_channels,
        padding_type="circular",
    ) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.total_channels = message_channels + visual_channels
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(self.total_channels, 1, 1).unsqueeze(1)
        all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.message_channels = message_channels
        self.visual_channels = visual_channels
        self.all_filters_batch = all_filters_batch
        self.rule = nn.Sequential(
            nn.Conv2d(3 * self.total_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.total_channels, kernel_size=1, bias=False),
        )
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold
        self.mass_conserving = mass_conserving
        self.padding_type = padding_type
        self.beta = beta

        assert mass_conserving in [
            "no",
            "normal",
            "cross_channel",
        ], "Bad mass_conserving option"


        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

    def forward(self, x, steps):
        seq = [x]

        for s in range(steps):
            x_padded = F.pad(x, (1, 1, 1, 1), self.padding_type)
            pre_life_mask = self.alive(x_padded, self.alive_threshold)

            delta = F.conv2d(
                F.pad(x, (1, 1, 1, 1), self.padding_type),
                self.all_filters_batch,
                groups=self.total_channels,
            )
            delta = self.rule(delta)
            # print(delta)
            if self.mass_conserving == "normal":
                affinity = delta[:, :self.visual_channels]
                q = x[:, :self.visual_channels]
                if self.visual_channels > 1:

                    q_next = cross_channel_mass_conserving_update(
                        beta=self.beta,
                        qs=q,
                        affinities=affinity,
                        padding_type=self.padding_type,
                    )
                else:
                    q_next = mass_conserving_update(
                        beta=self.beta,
                        q=q,
                        affinity=affinity,
                        padding_type=self.padding_type,
                    )

                x = torch.cat([q_next, x[:, self.visual_channels:] + delta[:, self.visual_channels:]], dim=1)
            elif self.mass_conserving == "cross_channel":
                raise NotImplementedError("cross_channel mass conservation removed for now")
            else:
                x = x + delta


            torch.clip_(x, -2, 2)
            post_life_mask = self.alive(
                F.pad(x, (1, 1, 1, 1), self.padding_type), self.alive_threshold
            )

            # life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            # if self.alive_threshold > 0 and (self.mass_conserving == "no"):
            #     x = x * life_mask

            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)
