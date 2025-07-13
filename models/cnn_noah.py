import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_sources=4, in_channels=1):
        """
        Args:
            n_sources (int): Number of sources to separate (e.g., vocals, drums, bass, other)
            in_channels (int): Number of input channels (1 for magnitude spectrogram)
        """
        super().__init__()
        self.n_sources = n_sources
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_sources, kernel_size=1)  # Output: [batch, n_sources, freq, time]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input spectrogram [batch, 1, freq_bins, time_frames]
        Returns:
            Tensor: Separated sources [batch, n_sources, freq_bins, time_frames]
        """
        return self.encoder(x)
