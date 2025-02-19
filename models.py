import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip


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


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 3):
        super().__init__()

        # Encoder path
        self.conv1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)  # 512 from up6 + 512 from conv4

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)  # 256 from up7 + 256 from conv3

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)  # 128 from up8 + 128 from conv2

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)  # 64 from up9 + 64 from conv1

        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)

        return conv10


class Autoencoder(nn.Module):
    """Autoencoder for pre-training"""

    def __init__(self, n_channels: int = 3):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dconv4 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConv(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConv(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, n_channels, kernel_size=1)

    def get_encoder(self):
        """Returns a copy of the encoder part"""
        encoder = nn.ModuleDict({
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'conv4': self.conv4,
            'pool': self.pool
        })
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        conv1 = self.conv1(x)
        x = self.pool(conv1)

        conv2 = self.conv2(x)
        x = self.pool(conv2)

        conv3 = self.conv3(x)
        x = self.pool(conv3)

        conv4 = self.conv4(x)
        x = self.pool(conv4)

        # Decoder
        x = self.upconv4(x)
        x = self.dconv4(x)

        x = self.upconv3(x)
        x = self.dconv3(x)

        x = self.upconv2(x)
        x = self.dconv2(x)

        x = self.upconv1(x)
        x = self.dconv1(x)

        x = self.final_conv(x)
        return torch.sigmoid(x)


class AESegmentation(nn.Module):
    """Segmentation model using pre-trained encoder"""

    def __init__(self, pretrained_encoder: nn.ModuleDict, n_classes: int = 3):
        super().__init__()

        # Encoder (pretrained and frozen)
        self.conv1 = pretrained_encoder['conv1']
        self.conv2 = pretrained_encoder['conv2']
        self.conv3 = pretrained_encoder['conv3']
        self.conv4 = pretrained_encoder['conv4']
        self.pool = pretrained_encoder['pool']

        # Freeze encoder weights
        for param in self.parameters():
            param.requires_grad = False

        # Decoder (trainable)
        self.up6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (frozen)
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        encoded = self.pool(conv4)

        # Decoder
        up6 = self.up6(encoded)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        return conv10


class CLIPSegmentation(nn.Module):
    """Segmentation model using frozen CLIP features"""

    def __init__(self, n_classes=3):
        super().__init__()

        # Load CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu')  # Can be moved to GPU later

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get CLIP's feature dimension (512 for ViT-B/32)
        self.feature_dim = self.clip_model.visual.output_dim

        # Decoder
        self.decoder = CLIPSegmentationDecoder(
            in_channels=self.feature_dim,
            n_classes=n_classes
        )

    def forward(self, x):
        # Get CLIP features
        with torch.no_grad():
            # Ensure input is in CLIP's expected format
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.clip_model.encode_image(x)  # [B, 512]

            # Reshape features to spatial form
            features = features.view(-1, self.feature_dim, 1, 1)  # [B, 512, 1, 1]
            features = features.expand(-1, -1, 7, 7)  # [B, 512, 7, 7]

        # Decode features to segmentation map
        out = self.decoder(features)

        # Resize to input resolution
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


class CLIPSegmentationDecoder(nn.Module):
    """Decoder for CLIP features to segmentation map"""

    def __init__(self, in_channels=512, n_classes=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        return x