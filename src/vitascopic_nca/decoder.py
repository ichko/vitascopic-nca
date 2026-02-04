from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_layers: int,
        hidden_dim: int,
        pooling_fn: Callable,
        padding_type: str,
    ):
        super().__init__()

        self.pooling_fn = pooling_fn
        self.padding_type = padding_type

        convs = []
        convs.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=0))
        for _layer in range(n_layers - 2):  # first and last are known
            convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=0))
        convs.append(nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=0))
        self.convs = nn.ModuleList(convs)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: (B, in_dim, 32, 32) binary image
        """
        h = x
        for idx, conv in enumerate(self.convs):
            h = F.pad(h, (1, 1, 1, 1), mode=self.padding_type)  # periodic/other BC
            h = conv(h)
            if idx < len(self.convs) - 1:
                h = self.activation(h)
        # h: (B, latent_dim, 32, 32)
        z = self.pooling_fn(h, dim=(2, 3))  # pool out the spatial dimensions
        return z  # (B, latent_dim)


if __name__ == "__main__":
    batches = 2

    x = torch.randn(batches, 1, 32, 32)

    print("before decoder:")
    print(x.shape)
    decoder = Decoder(
        in_dim=1,
        latent_dim=8,
        hidden_dim=8,
        n_layers=4,
        pooling_fn=torch.amax,
        padding_type="circular",
    )
    z = decoder(x)
    print("after decoder:")
    print(z.shape)  # (B, latent_dim)
