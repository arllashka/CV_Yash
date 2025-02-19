import torch
import torch.nn as nn
import torch.nn.functional as F


class PointSegmentationModel(nn.Module):
    """UNet-based architecture modified for point-prompted segmentation"""

    def __init__(self, n_channels: int = 3):
        super().__init__()

        # Initial convolution for point heatmap
        self.point_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Modified encoder for image (adding point features)
        self.conv1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(128, 128)  # 128 input = 64 from image + 64 from point
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)  # Output is binary mask

    def forward(self, x: torch.Tensor, point_heatmap: torch.Tensor) -> torch.Tensor:
        # Process point heatmap
        p1 = self.point_conv(point_heatmap)

        # Encoding path with point features
        conv1 = self.conv1(x)
        # Concatenate image and point features
        conv1 = torch.cat([conv1, p1], dim=1)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)

        conv5 = self.conv5(pool4)

        # Decoding path
        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([self.conv1(x), up9], dim=1)  # Use original image features
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)

        return torch.sigmoid(conv10).squeeze(1)  # Remove channel dimension to match target shape


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)