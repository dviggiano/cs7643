# modeling/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


#################### Encoder/Decoder CNN ####################


class VanillaCNN(nn.Module):
    """
    Encoder-decoder CNN with two blocks
    mix (1xFxT) -> 4 stems (4xFxT)
    """
    def __init__(self, in_channels=1, base_filters=32, output_channels=4):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        )
        # decode: upsample -> ReLU -> output conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



#################### U NET ####################
class SimpleUNet(nn.Module):
    """
    Symmetrical U-Net with 3 encoder/decoder levels and skip connections.
    Input:  (B, 1, F, T)
    Output: (B, 4, F, T)
    """
    def __init__(self, in_channels=1, base_filters=32, output_channels=4, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2  # preserve spatial dimensions

        # --- Encoder blocks ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

        # --- Decoder blocks ---
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

        # --- Output ---
        self.out_conv = nn.Conv2d(base_filters, output_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))

        # Decoder
        x = self.upconv3(x4)
        x = F.pad(x, self._get_pad(x, x3))
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.upconv2(x)
        x = F.pad(x, self._get_pad(x, x2))
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.upconv1(x)
        x = F.pad(x, self._get_pad(x, x1))
        x = self.dec1(torch.cat([x, x1], dim=1))

        return torch.sigmoid(self.out_conv(x))

    def _get_pad(self, upsampled, skip):
        """Returns padding tuple to match shape (right, left, bottom, top)."""
        diffY = skip.size(2) - upsampled.size(2)
        diffX = skip.size(3) - upsampled.size(3)
        return [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]