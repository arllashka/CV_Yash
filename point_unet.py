import torch
import torch.nn as nn
import torch.nn.functional as F


class PointUNet(nn.Module):
    """UNet architecture modified for point-based segmentation"""

    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()

        # Point encoder branch
        self.point_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Image encoder path
        self.conv1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path with point features
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024 + 32, 512)  # 512 from up6 + 512 from conv4 + 32 from point features

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512 + 32, 256)  # 256 from up7 + 256 from conv3 + 32 from point features

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256 + 32, 128)  # 128 from up8 + 128 from conv2 + 32 from point features

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128 + 32, 64)  # 64 from up9 + 64 from conv1 + 32 from point features

        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, point_heatmap):
        # Process point heatmap
        point_features = self.point_encoder(point_heatmap)

        # Encoding path
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)

        conv5 = self.conv5(pool4)

        # Resize point features for each decoder stage
        point_feat6 = F.interpolate(point_features, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        point_feat7 = F.interpolate(point_features, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        point_feat8 = F.interpolate(point_features, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        point_feat9 = F.interpolate(point_features, size=conv1.shape[2:], mode='bilinear', align_corners=False)

        # Decoding path with point feature concatenation
        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6, point_feat6], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7, point_feat7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8, point_feat8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9, point_feat9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)

        return conv10


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )