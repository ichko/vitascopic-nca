import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


PADDING_TYPE = "circular"  # e.g. 'circular', 'reflect', 'replicate'



def shannon_entropy(values: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """Shannon entropy of a 1D tensor via histogram binning."""
    hist = torch.histc(values, bins=num_bins, min=values.min(), max=values.max())
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -(prob * torch.log(prob)).sum()


def global_entropy_over_time(states: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """Global entropy (all channels pooled) over time."""
    T = states.shape[0]
    entropies = []
    for t in range(T):
        x = states[t].reshape(-1)
        entropies.append(shannon_entropy(x, num_bins=num_bins))
    return torch.stack(entropies)


def per_channel_entropy_over_time(states: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """Entropy per channel over time.

    Returns:
        entropies: (T, C)
    """
    T, C = states.shape[:2]
    entropies = torch.zeros((T, C), device=states.device)

    for t in range(T):
        for c in range(C):
            entropies[t, c] = shannon_entropy(states[t, c].reshape(-1), num_bins=num_bins)

    return entropies


# ----------------------------
# Sliding-window entropy (final timestep, vectorized)
# ----------------------------

def sliding_window_entropy_final(
    final_state: torch.Tensor,
    window_size: int = 9,
    num_bins: int = 32,
    padding_type: str = PADDING_TYPE,
) -> torch.Tensor:
    """
    Vectorized local entropy map using torch.unfold.

    Args:
        final_state: (C, H, W)
    Returns:
        entropy_map: (H, W)
    """
    assert window_size % 2 == 1, "window_size must be odd"

    C, H, W = final_state.shape
    pad = window_size // 2

    # Pad spatial dimensions only
    x = F.pad(final_state, (pad, pad, pad, pad), mode=padding_type)

    # Extract sliding windows
    # Result: (C, H, W, window_size, window_size)
    patches = x.unfold(1, window_size, 1).unfold(2, window_size, 1)

    # Reshape to (H, W, C * window_size * window_size)
    patches = patches.permute(1, 2, 0, 3, 4)
    patches = patches.reshape(H, W, -1)

    # Compute entropy per spatial location
    entropy_map = torch.zeros((H, W), device=final_state.device)

    for i in range(H):
        for j in range(W):
            entropy_map[i, j] = shannon_entropy(patches[i, j], num_bins=num_bins)

    return entropy_map


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_global_entropies_over_time(states: torch.Tensor):
    """Plot global and per-channel entropy trajectories."""
    H_global = global_entropy_over_time(states).cpu().numpy()
    H_channels = per_channel_entropy_over_time(states).cpu().numpy()

    T, C = H_channels.shape

    plt.figure(figsize=(7, 4))
    for c in range(C):
        plt.plot(H_channels[:, c], alpha=0.6, label=f'Channel {c}')

    plt.plot(H_global, color='black', linewidth=2, label='Global entropy')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.title('Shannon entropy over time')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_final_entropy_map(entropy_map: torch.Tensor):
    """Plot local entropy heatmap for final timestep."""
    plt.figure(figsize=(5, 4))
    plt.imshow(entropy_map.cpu().numpy(), cmap='inferno')
    plt.colorbar(label='Local entropy')
    plt.title('Final-step local entropy map')
    plt.tight_layout()
    plt.show()


# ----------------------------
# High-level convenience API
# ----------------------------

def analyze_nca_run(
    states: torch.Tensor,
    window_size: int = 9,
    padding_type: str = PADDING_TYPE,
):
    """Full entropy analysis for a single NCA rollout."""
    assert states.dim() == 4, "Expected (T, C, H, W) tensor"

    plot_global_entropies_over_time(states)

    final_state = states[-1]
    entropy_map = sliding_window_entropy_final(
        final_state,
        window_size=window_size,
        padding_type=padding_type,
    )
    plot_final_entropy_map(entropy_map)


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    T, C, H, W = 32, 9, 64, 64
    dummy_states = torch.randn(T, C, H, W)

    analyze_nca_run(dummy_states, window_size=9)
