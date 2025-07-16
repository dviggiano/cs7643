import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_sources=4, in_channels=1, base_channels=16):
        """
        Args:
            n_sources (int): Number of sources to separate (e.g., vocals, drums, bass, other)
            in_channels (int): Number of input channels (1 for magnitude spectrogram)
            base_channels (int): Number of channels for the first conv layer
        """
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU()
        )

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # Final output layer
        self.final = nn.Conv2d(base_channels, n_sources, 1)

    def center_crop(self, enc_feat, dec_feat):
        """
        Center-crop enc_feat to match the spatial size of dec_feat
        Args:
            enc_feat: [B, C, H1, W1]
            dec_feat: [B, C, H2, W2]
        Returns:
            Cropped enc_feat: [B, C, H2, W2]
        """
        _, _, h1, w1 = enc_feat.shape
        _, _, h2, w2 = dec_feat.shape
        dh = (h1 - h2) // 2
        dw = (w1 - w2) // 2
        return enc_feat[:, :, dh:dh+h2, dw:dw+w2]

    def forward(self, x):
        """
        Args:
            x (Tensor): Input spectrogram [batch, 1, freq_bins, time_frames]
        Returns:
            Tensor: Separated sources [batch, n_sources, freq_bins, time_frames]
        """
        # Encoder
        e1 = self.enc1(x)  # [B, C, F, T]
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        e3_cropped = self.center_crop(e3, d3)
        d3 = torch.cat([d3, e3_cropped], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_cropped = self.center_crop(e2, d2)
        d2 = torch.cat([d2, e2_cropped], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_cropped = self.center_crop(e1, d1)
        d1 = torch.cat([d1, e1_cropped], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
