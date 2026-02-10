from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
import torch


def impact_frames(x, ts, ns):
    for t, n in reversed(list(zip(ts, ns))):
        frame = x[:, t : t + 1]
        repeated = frame.repeat(1, n, 1, 1, 1)
        x = torch.cat([x[:, : t + 1], repeated, x[:, t + 1 :]], dim=1)

    return x


def image_row(frame_batches, columns):
    return pn.Row(
        *[
            pn.pane.HTML(
                media.show_images(
                    frame_batch[:, 0],
                    columns=columns,
                    width=120,
                    height=120,
                    cmap="viridis",
                    return_html=True,
                )
            )
            for frame_batch in frame_batches
        ]
    )


def make_sobel_kernels(size: int):
    assert size % 2 == 1 and size >= 3, "sobel_size must be odd and >= 3"

    # Binomial coefficients for smoothing
    def binomial(n):
        row = [1]
        for _ in range(n):
            row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
        return torch.tensor(row)

    smooth_1d = binomial(size - 1)
    deriv_1d = torch.zeros(size)
    deriv_1d[0] = -1
    deriv_1d[-1] = 1

    smooth_1d = smooth_1d / smooth_1d.sum()

    sobel_x = torch.outer(smooth_1d, deriv_1d)
    sobel_y = torch.outer(deriv_1d, smooth_1d)

    identity = torch.zeros(size, size)
    identity[size // 2, size // 2] = 1.0

    return identity, sobel_x, sobel_y


def tensor_summary(T):
    if type(T) is not torch.Tensor:
        T = torch.tensor([v for t in T for v in t.detach().cpu().flatten()])

    T = T.detach().cpu().flatten().numpy()
    return (
        f"[{np.min(T):.4f}, {np.max(T):.4f}] μ = {np.mean(T):.4f}, σ = {np.std(T):.4f}"
    )


def plot_bars(batch1, batch2):
    B, D = batch1.shape
    df1 = pd.DataFrame(
        {
            "batch": np.repeat(np.arange(B), D),
            "dim": np.tile(np.arange(D), B),
            "value": batch1.reshape(-1),
            "source": "batch1",
        }
    )
    df2 = pd.DataFrame(
        {
            "batch": np.repeat(np.arange(B), D),
            "dim": np.tile(np.arange(D), B),
            "value": batch2.reshape(-1),
            "source": "batch2",
        }
    )

    df = pd.concat([df1, df2], axis=0)

    g = sns.catplot(
        data=df,
        kind="bar",
        x="dim",
        y="value",
        hue="source",
        col="batch",
        sharey=True,
        height=1.8,
        aspect=0.9,
        alpha=0.8,  # transparency
        dodge=False,  # stack on top of each other
        legend=False,
    )

    # Clean up axes
    for ax in g.axes.flat:
        ax.set_ylabel(None)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.tick_params(bottom=False)

    g.set_titles("")

    fig = g.figure
    plt.close(fig)

    return pn.pane.Matplotlib(fig, format="svg", width=32 * D, height=70, tight=True)


def sequence_batch_to_html_gifs(
    tensor, width, height, return_html=False, columns=8, fps=20
):
    tensor = tensor.detach().cpu().numpy()
    tensor = media.to_rgb(tensor, cmap="viridis", vmin=0, vmax=1)
    tensor = tensor[:, :, :, :, :3]

    return media.show_videos(
        tensor,
        titles=[f"#{i}" for i in range(tensor.shape[0])],
        fps=fps,
        codec="gif",
        columns=columns,
        width=width,
        height=height,
        return_html=return_html,
    )


def export_neural_ca_to_json(
    model,
    out_path: Union[str, Path],
    model_name: str = "neural_ca",
) -> None:
    """
    Export a trained NeuralCA model to a JSON file suitable for loading in JS.

    The JSON schema is compatible with the browser-side NCA viewer, roughly:

    {
      "model_type": "NeuralCA",
      "model_name": "...",
      "channels": int,
      "hidden_channels": int,
      "alive_threshold": float,
      "fire_rate": float,
      "padding_type": "circular",
      "rule": {
        "conv1": {
          "weight": [[...],[...]],  # [hidden_channels][3*channels]
          "bias":   [...]
        },
        "conv2": {
          "weight": [[...],[...]]   # [channels][hidden_channels]
        }
      }
    }
    """
    model = model.to("cpu")
    model.eval()

    # Access the two conv layers from the Sequential rule defined in NeuralCA
    conv1 = model.rule[0]  # nn.Conv2d(3 * channels, hidden_channels, 1)
    conv2 = model.rule[2]  # nn.Conv2d(hidden_channels, channels, 1), bias=False

    # Sanity checks
    assert hasattr(conv1, "weight") and hasattr(conv1, "bias")
    assert hasattr(conv2, "weight")
    assert conv1.kernel_size == (1, 1)
    assert conv2.kernel_size == (1, 1)

    hidden_channels, three_c, _, _ = conv1.weight.shape
    channels, hidden_c2, _, _ = conv2.weight.shape

    assert three_c == 3 * model.channels, "conv1 in_channels must be 3 * channels"
    assert hidden_c2 == hidden_channels, "conv2 in_channels must equal hidden_channels"

    data = {
        "model_type": "NeuralCA",
        "model_name": model_name,
        "channels": int(model.channels),
        "hidden_channels": int(hidden_channels),
        "alive_threshold": float(model.alive_threshold),
        "fire_rate": float(getattr(model, "fire_rate", 1.0)),
        "padding_type": str(getattr(model, "padding_type", "circular")),
        "rule": {
            "conv1": {
                "weight": conv1.weight.detach()
                .cpu()
                .numpy()
                .reshape(hidden_channels, three_c)
                .tolist(),
                "bias": conv1.bias.detach().cpu().numpy().tolist(),
            },
            "conv2": {
                "weight": conv2.weight.detach()
                .cpu()
                .numpy()
                .reshape(channels, hidden_channels)
                .tolist(),
            },
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
