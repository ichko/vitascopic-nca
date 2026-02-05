import torch


class Stimuli:
    """Class to represent stimuli for the NCA. For simplicity, we just use random noise as stimuli."""

    def __init__(self, initial_state):
        # self.initial_state = initial_state # batch_size, channels, H, W
        N = 1
        R = 8
        # add N circles of radius R to the initial state as stimuli
        self.stimuli = torch.zeros_like(initial_state[:, :1, :, :])  # only 1 channel for stimuli
        self.stimuli_centers = torch.zeros_like(initial_state[:, :1, :, :])
        for _ in range(N):
            # with peridoic boundary conditions
            center_x = torch.randint(0, initial_state.shape[3], (1,)).item()
            center_y = torch.randint(0, initial_state.shape[2], (1,)).item()

            y, x = torch.meshgrid(
                torch.arange(initial_state.shape[2]), torch.arange(initial_state.shape[3])
            )

            # with periodic boundary conditions
            dist_x = torch.min(torch.abs(x - center_x), torch.abs(x - center_x + initial_state.shape[3]))
            dist_y = torch.min(torch.abs(y - center_y), torch.abs(y - center_y + initial_state.shape[2]))
            mask = (dist_x) ** 2 + (dist_y) ** 2 <= R**2

            center_mask = (dist_x) ** 2 + (dist_y) ** 2 <= 2**2

            self.stimuli_centers[:, :, center_mask] = 1.  # set stimulus center value to 1

            self.stimuli[:, :, mask] = 1.  # set stimulus value to 0.5

        self.stimuli = self.stimuli.squeeze(1)  # shape (batch_size, H, W)
        self.stimuli = self.stimuli.to(initial_state.device)

        self.stimuli_centers = self.stimuli_centers.squeeze(1)  # shape (batch_size, H, W)
        self.stimuli_centers = self.stimuli_centers.to(initial_state.device)

    def add_stimuli_noise(self, final_frame):
        return torch.where(self.stimuli.unsqueeze(1) < 0.5, torch.zeros_like(final_frame), final_frame)
    
    def add_stimuli(self, state):
        # state[:, 1, :, :] =  state[:, 1, :, :] + self.stimuli  # add stimuli to the second channel
        state[:, 1, :, :] =  state[:, 1, :, :] + self.stimuli_centers  # add stimuli to the second channel

        return state
