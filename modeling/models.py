# modeling/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


#################### Econder/Decoder CNN ####################


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
    Lightweight U-Net with encoder block, decoder block, and a skip connection.
    mix (1xFxT) -> 4 stems (4xFxT)
    """
    def __init__(self, in_channels=1, base_filters=32, output_channels=4):
        super().__init__()

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv = nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, kernel_size=3, stride=1, padding=1)
        )

        # Temporal smoothing
        self.smooth = nn.Conv2d(1, 1, (1,5), padding=(0,2))


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
        mask = self.smooth(out)
        return torch.sigmoid(mask)
