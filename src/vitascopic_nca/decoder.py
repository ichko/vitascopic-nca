import torch
import torch.nn as nn
from typing import Callable




class Decoder(nn.Module):
    def __init__(self, latent_dim : int, pooling_fn : Callable):
        super().__init__()

        self.pooling_fn = pooling_fn

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x,):
        """
        x: (B, 1, 32, 32) binary image
        """
        h = self.conv(x)                         # (B, latent_dim, 32, 32)
        z = self.pooling_fn(h, dim=(2, 3))       # configurable pooling
        return z                                 # (B, latent_dim)
    

if __name__ == "__main__":
    batches = 2
    x = torch.randn(batches, 1, 32, 32)
    decoder = Decoder(latent_dim=16, pooling_fn=torch.amax)
    z = decoder(x)
    print(z.shape)  # (B, latent_dim)