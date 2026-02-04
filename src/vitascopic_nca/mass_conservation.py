import torch
import torch.nn.functional as F

EPS = 1e-8


def mass_conserving_update(beta, q, affinity, padding_type):
    """
    Mass-conserving update for a single-channel density field `q` using an
    affinity field `affinity` as described in the MaCE rule. Only channel 0 is
    expected to be passed in.
    """
    weights = torch.exp(beta * affinity)
    kernel = torch.ones((1, 1, 3, 3), device=q.device, dtype=q.dtype)

    weights_padded = F.pad(weights, (1, 1, 1, 1), mode=padding_type)
    Z = F.conv2d(weights_padded, kernel, padding=0)

    q_over_Z = q / (Z + EPS)
    q_over_Z_padded = F.pad(q_over_Z, (1, 1, 1, 1), mode=padding_type)
    incoming = F.conv2d(q_over_Z_padded, kernel, padding=0)

    q_next = weights * incoming
    return q_next