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

from vitascopic_nca.base_trainer import BaseTrainer
from vitascopic_nca.nca import NeuralCA


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
                    width=100,
                    height=100,
                    cmap="viridis",
                    return_html=True,
                )
            )
            for frame_batch in frame_batches
        ]
    )


def tensor_summary(T):
    if type(T) is not torch.Tensor:
        T = torch.tensor([v for t in T for v in t.detach().cpu().flatten()])

    T = T.detach().cpu().flatten().numpy()
    return (
        f"[{np.min(T):.4f}, {np.max(T):.4f}] μ = {np.mean(T):.4f}, σ = {np.std(T):.4f}"
    )


def from_batch_to_DNA_string(batch):
    # batch shape (B, D)
    characters = ["A", "C", "G", "T"]
    strings = []

    reshaped = batch.view(batch.shape[0], -1, 4)  # reshape to (B, D//4, 4)

    for sample in reshaped:
        string = ""
        for group in sample:
            idx = torch.argmax(group).item()  # get index of max value in the group of 4
            string += characters[idx]
        strings.append(string)
    return strings

def plot_bars(batch1, batch2, loss_type):
    if loss_type == "mse" or loss_type == "clf":

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
    
        return pn.pane.Matplotlib(fig, format="svg", width=50 * D, height=110, tight=True)

    elif loss_type == "DNA":

        strings1 = from_batch_to_DNA_string(batch1)
        strings2 = from_batch_to_DNA_string(batch2)

        # show side by side in a table
        table = "<table><tr><th>Batch</th><th>Input Msg</th><th>Output Msg</th></tr>"
        for i, (s1, s2) in enumerate(zip(strings1, strings2)):
            table += f"<tr><td>{i}</td><td>{s1}</td><td>{s2}</td></tr>"
        table += "</table>"
        return pn.pane.HTML(table)

    else:
        return pn.pane.Markdown("No bar plot for loss type: " + loss_type)


def sequence_batch_to_html_gifs(
    tensor, width, height, return_html=False, columns=8, fps=20
):
    tensor = tensor.detach().cpu().numpy()
    if tensor.shape[2] == 1:
        tensor = media.to_rgb(tensor, cmap="viridis", vmin=0, vmax=1)
    else:
        # If there are more than 3 channels, just take the first 3 for visualization
        tensor = tensor[:, :, :3, :, :]
        # move channels to the end for media.show_videos, which expects (B, t, H, W, C)
        tensor = np.transpose(tensor, (0, 1, 3, 4, 2))
        # If there are exactly 3 channels, keep them as is. If there are fewer than 3 channels, the above line will just take what's available.
        # make into RGB if it's not already
        print("tensor.shape", tensor.shape)    
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
    model: NeuralCA,
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
        "mass_conserving": str(getattr(model, "mass_conserving", "no")),
        "beta": float(getattr(model, "beta", 1.0)),
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

    print(f"Exported NeuralCA model to {out_path.resolve()}")


def export_checkpoint_to_json(
    checkpoint_dir: Union[str, Path],
    out_path: Union[str, Path],
    model_name: str | None = None,
) -> None:
    """
    Load a trainer checkpoint (from a directory containing 'self.pkl') and export
    its NCA model to JSON.

    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., containing 'self.pkl')
        out_path: Where to write the JSON file
        model_name: Optional name for the model. If None, uses the checkpoint directory name.

    Example:
        >>> export_checkpoint_to_json(
        ...     checkpoint_dir="./checkpoints/20240101_120000_abc12345",
        ...     out_path="./docs/models/my_nca.json",
        ...     model_name="trained_nca_v1"
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load the trainer from pickle
    trainer = BaseTrainer.load_trainer(checkpoint_dir)

    # Load the latest NCA weights
    trainer.load_last_checkpoint()

    # Extract the NCA model
    nca = trainer.nca

    # Use checkpoint dir name as default model name
    if model_name is None:
        model_name = checkpoint_dir.name

    # Export to JSON
    export_neural_ca_to_json(nca, out_path, model_name=model_name)
