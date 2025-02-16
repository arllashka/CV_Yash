import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
        self.conv7 = DoubleConv(512, 256)   # 256 from up7 + 256 from conv3
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)   # 128 from up8 + 128 from conv2
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)    # 64 from up9 + 64 from conv1
        
        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding path
        conv1 = self.conv1(x)       # 64
        pool1 = self.pool(conv1)    # 64
        
        conv2 = self.conv2(pool1)   # 128
        pool2 = self.pool(conv2)    # 128
        
        conv3 = self.conv3(pool2)   # 256
        pool3 = self.pool(conv3)    # 256
        
        conv4 = self.conv4(pool3)   # 512
        pool4 = self.pool(conv4)    # 512
        
        conv5 = self.conv5(pool4)   # 1024
        
        # Decoding path
        up6 = self.up6(conv5)       # 512
        merge6 = torch.cat([conv4, up6], dim=1)  # 512 + 512 = 1024
        conv6 = self.conv6(merge6)  # 512
        
        up7 = self.up7(conv6)       # 256
        merge7 = torch.cat([conv3, up7], dim=1)  # 256 + 256 = 512
        conv7 = self.conv7(merge7)  # 256
        
        up8 = self.up8(conv7)       # 128
        merge8 = torch.cat([conv2, up8], dim=1)  # 128 + 128 = 256
        conv8 = self.conv8(merge8)  # 128
        
        up9 = self.up9(conv8)       # 64
        merge9 = torch.cat([conv1, up9], dim=1)  # 64 + 64 = 128
        conv9 = self.conv9(merge9)  # 64
        
        conv10 = self.conv10(conv9) # n_classes
        
        return conv10
        
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)