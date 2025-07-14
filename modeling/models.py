# modeling/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaCNN(nn.Module):
    """
    Simple encoder-decoder CNN that maps
    mix (1xFxT) → 4 stems (4xFxT).
    """
    def __init__(self, in_channels=1, base_filters=32, output_channels=4):
        super().__init__()
        # encode: conv -> ReLU -> downsample
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # decode: upsample -> ReLU -> output conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters,
                               base_filters // 2,
                               kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2,
                      output_channels,
                      kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



#################### U NET ####################

class SimpleUNet(nn.Module):
    """
    Lightweight U-Net with one encoder block, one decoder block, and a skip connection.
    Input:  (B, 1, F, T)
    Output: (B, 4, F, T)
    """
    def __init__(self, in_channels=1, base_filters=32, output_channels=4):
        super().__init__()

        # Encoder block
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

        # Decoder block
        self.upconv = nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x_enc = self.enc_conv(x)       # (B, base_filters, F, T)
        x_down = self.pool(x_enc)      # (B, base_filters, F/2, T/2)

        # Decoder
        x_up = self.upconv(x_down)     # (B, base_filters, F, T)

        # Match shapes in case of odd dims (center crop encoder output)
        if x_up.shape[-2:] != x_enc.shape[-2:]:
            x_up = F.pad(x_up, (0, x_enc.shape[-1] - x_up.shape[-1],
                                0, x_enc.shape[-2] - x_up.shape[-2]))

        # Concatenate skip connection
        x_cat = torch.cat([x_enc, x_up], dim=1)  # (B, base_filters*2, F, T)

        out = self.dec_conv(x_cat)     # (B, output_channels, F, T)
        return out
