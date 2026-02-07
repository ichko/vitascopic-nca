import torch


class Stimuli:
    """Class to represent stimuli for the NCA. For simplicity, we just use random noise as stimuli."""

    def __init__(self, initial_state):
        # self.initial_state = initial_state # batch_size, channels, H, W
        N = 1
        R = 10
        sigma = R / 2.0
        device = initial_state.device
        # add N Gaussian blobs (periodic boundaries) to the initial state as stimuli

        # only 1 channel for stimuli
        self.stimuli = torch.zeros_like(initial_state[:, :1, :, :]).to(device)
        self.stimuli_centers = torch.zeros_like(initial_state[:, :1, :, :]).to(device)
        for _ in range(N):
            # with periodic boundary conditions
            center_x = torch.randint(0, initial_state.shape[3], (1,)).item()
            center_y = torch.randint(0, initial_state.shape[2], (1,)).item()

            y, x = torch.meshgrid(
                torch.arange(initial_state.shape[2]),
                torch.arange(initial_state.shape[3]),
                indexing="ij",
            )

            # with periodic boundary conditions
            dist_x = torch.min(
                torch.abs(x - center_x),
                torch.abs(x - center_x + initial_state.shape[3]),
            )
            dist_y = torch.min(
                torch.abs(y - center_y),
                torch.abs(y - center_y + initial_state.shape[2]),
            )

            r2 = dist_x**2 + dist_y**2
            gaussian = torch.exp(-r2.float() / (2 * sigma**2)).to(device)

            center_mask = r2 <= 2**2

            # set stimulus center value to 1
            self.stimuli_centers[:, :, center_mask] = 1.0
            self.stimuli = self.stimuli + gaussian  # broadcast over batch

        self.stimuli = self.stimuli.squeeze(1)  # shape (batch_size, H, W)
        self.stimuli = self.stimuli.to(initial_state.device)

        # shape (batch_size, H, W)
        self.stimuli_centers = self.stimuli_centers.squeeze(1)
        self.stimuli_centers = self.stimuli_centers.to(initial_state.device)

    def add_stimuli_noise(self, final_frame):
        return final_frame * (self.stimuli.unsqueeze(1))

    def add_stimuli(self, state):
        # state[:, 1, :, :] =  state[:, 1, :, :] + self.stimuli  # add stimuli to the second channel
        state[:, 1, :, :] = (
            state[:, 1, :, :] + self.stimuli_centers
        )  # add stimuli to the second channel

        return state
