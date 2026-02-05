import torch
import torch.nn.functional as F


EPS = 1e-8

beta = 5.0


def mass_conserving_update(q, affinity, padding_type):
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


def cross_channel_mass_conserving_update(qs, affinities, padding_type):
	"""
	Mass-conserving update for a single-channel density field `q` using an
	affinity field `affinity` as described in the MaCE rule. Multiple channels is
	expected to be passed in.
	"""

	# normal
	channel_count = qs.shape[1]
	weights = torch.exp(beta * affinities)
	kernel = torch.ones((channel_count, 1, 3, 3), device=qs.device, dtype=qs.dtype)

	weights_padded = F.pad(weights, (1, 1, 1, 1), mode=padding_type)
	Z = F.conv2d(weights_padded, kernel, padding=0, groups=channel_count)

	q_over_Z = qs / (Z + EPS)
	q_over_Z_padded = F.pad(q_over_Z, (1, 1, 1, 1), mode=padding_type)
	incoming = F.conv2d(q_over_Z_padded, kernel, padding=0, groups=channel_count)

	q_next = weights * incoming

	# cross channel
	weights_cross = torch.exp(beta * affinities)
	Z_cross = weights_cross.sum(dim=1, keepdim=True) 

	q_over_Z_cross = q_next / (Z_cross + EPS)

	q_next_w_cross = (q_over_Z_cross * weights_cross)	

	print("q_over_Z_cross shape:", q_next_w_cross.shape)
	return q_next_w_cross


if __name__ == "__main__":
	example_q = torch.randn((1, 1, 8, 8))
	example_affinity = torch.randn((1, 1, 8, 8))

	print("q shape before update:")
	print(example_q.shape)
	updated_q = mass_conserving_update(example_q, example_affinity, padding_type="circular")
	print("q shape after update:")
	print(updated_q.shape)

	example_qs = torch.randn((1, 2, 8, 8))
	example_affinities = torch.randn((1, 2, 8, 8))
	print("qs shape before update:")
	print(example_qs.shape)
	updated_qs = cross_channel_mass_conserving_update(example_qs, example_affinities, padding_type="circular")
	print("qs shape after update:")
	print(updated_qs.shape)