from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch

from .nca import NeuralCA


def impact_frames(inp):
    # TODO: implement or remove if unused
    raise NotImplementedError("repeat_dim is not implemented")


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

