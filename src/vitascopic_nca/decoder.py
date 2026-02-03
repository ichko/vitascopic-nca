import torch
import torch.nn as nn
from typing import Callable


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_layers: int,
        hidden_dim: int,
        pooling_fn: Callable,
    ):
        super().__init__()

        self.pooling_fn = pooling_fn

        layers = []

        layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        for _layer in range(n_layers - 2): # first and last we know what are 
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        """
        x: (B, in_dim, 32, 32) binary image
        """
        h = self.conv(x)  # (B, latent_dim, 32, 32)
        z = self.pooling_fn(h, dim=(2, 3))  # pool out the spatial dimensions
        return z  # (B, latent_dim)


if __name__ == "__main__":
    batches = 2

    x = torch.randn(batches, 1, 32, 32)

    print("before decoder:")
    print(x.shape)
    decoder = Decoder(
        in_dim=1, latent_dim=8, hidden_dim=8, n_layers=4, pooling_fn=torch.amax
    )
    z = decoder(x)
    print("after decoder:")
    print(z.shape)  # (B, latent_dim)
