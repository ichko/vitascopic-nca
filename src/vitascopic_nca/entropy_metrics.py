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


def shannon_entropy_2d(values: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """Joint Shannon entropy for pairs of variables.

    Expects the last dimension to have size 2 (e.g., (..., 2) or (2, N)). Uses a
    2D histogram to estimate the joint distribution.
    """
    if values.dim() == 2 and values.shape[0] == 2 and values.shape[1] != 2:
        values = values.t()

    assert values.shape[-1] == 2, "Expected values with last dimension = 2 for joint entropy"

    flat = values.reshape(-1, 2)
    x = flat[:, 0].detach().cpu().numpy()
    y = flat[:, 1].detach().cpu().numpy()

    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return torch.tensor(-(prob * np.log(prob)).sum(), device=values.device, dtype=values.dtype)


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


def pairwise_channel_entropy_over_time(
    states: torch.Tensor,
    channel_pairs: tuple[tuple[int, int], ...] | None = None,
    num_bins: int = 64,
) -> torch.Tensor:
    """Joint entropy for selected channel pairs over time.

    Args:
        states: (T, C, H, W)
        channel_pairs: tuple of (i, j) channel indices. If None, defaults to ((0, 1),)
            when C >= 2, otherwise returns an empty tensor.
    Returns:
        entropies: (T, P) where P = len(channel_pairs)
    """
    T, C = states.shape[:2]

    if channel_pairs is None:
        if C < 2:
            return torch.empty((T, 0), device=states.device)
        channel_pairs = ((0, 1),)

    entropies = torch.zeros((T, len(channel_pairs)), device=states.device)

    for t in range(T):
        for idx, (i, j) in enumerate(channel_pairs):
            pair_vals = torch.stack((states[t, i], states[t, j]), dim=-1)  # (H, W, 2)
            entropies[t, idx] = shannon_entropy_2d(pair_vals, num_bins=num_bins)

    return entropies


def spatial_mass_entropy_over_time(states: torch.Tensor, clamp_min: float = 0.0) -> torch.Tensor:
    """Entropy of spatial mass distribution per channel over time.

    For each channel and timestep, normalize the HxW field to a probability map
    and compute -sum(p * log p). A uniform spread gives high entropy; a
    concentrated blob gives low entropy.

    Args:
        states: (T, C, H, W)
        clamp_min: values below this are clipped before normalization to avoid
            negatives. Use 0.0 for nonnegative mass fields.
    Returns:
        entropies: (T, C)
    """
    T, C = states.shape[:2]
    entropies = torch.zeros((T, C), device=states.device, dtype=states.dtype)

    eps = torch.finfo(states.dtype).eps

    for t in range(T):
        for c in range(C):
            field = states[t, c]
            field = torch.clamp(field, min=clamp_min)
            total = field.sum()
            if total <= eps:
                entropies[t, c] = torch.tensor(0.0, device=states.device, dtype=states.dtype)
                continue
            p = field / (total + eps)
            entropies[t, c] = -(p * (p + eps).log()).sum()

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


def plot_pairwise_entropies_over_time(
    states: torch.Tensor,
    channel_pairs: tuple[tuple[int, int], ...] | None = None,
    num_bins: int = 64,
):
    """Plot joint entropy trajectories for selected channel pairs."""
    H_pairs = pairwise_channel_entropy_over_time(states, channel_pairs=channel_pairs, num_bins=num_bins)
    if H_pairs.numel() == 0:
        return

    H_pairs_np = H_pairs.cpu().numpy()
    T, P = H_pairs_np.shape

    # Ensure we have explicit pairs for labeling
    if channel_pairs is None:
        C = states.shape[1]
        channel_pairs = ((0, 1),) if C >= 2 else tuple()

    plt.figure(figsize=(7, 4))
    for idx, pair in enumerate(channel_pairs):
        plt.plot(H_pairs_np[:, idx], label=f"Channels {pair[0]} & {pair[1]}", alpha=0.7)

    plt.xlabel('Time step')
    plt.ylabel('Joint entropy')
    plt.title('Pairwise channel joint entropy over time')
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
    channel_pairs: tuple[tuple[int, int], ...] | None = None,
    show_mass_entropy: bool = False,
    padding_type: str = PADDING_TYPE,
):
    """Full entropy analysis for a single NCA rollout."""
    assert states.dim() == 4, "Expected (T, C, H, W) tensor"

    plot_global_entropies_over_time(states)
    plot_pairwise_entropies_over_time(states, channel_pairs=channel_pairs)

    if show_mass_entropy:
        H_mass = spatial_mass_entropy_over_time(states)
        plt.figure(figsize=(7, 4))
        for c in range(H_mass.shape[1]):
            plt.plot(H_mass[:, c].cpu().numpy(), label=f"Channel {c}", alpha=0.6)
        plt.xlabel('Time step')
        plt.ylabel('Spatial mass entropy')
        plt.title('Spatial mass entropy over time')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

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
