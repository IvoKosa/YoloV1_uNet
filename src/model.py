import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Number of channels in the input feature map from the previous decoder layer.
            skip_channels: Number of channels from the corresponding encoder feature map (skip connection).
            out_channels: Number of output channels after upsampling and convolution.
        """
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class YOLO_UNet(nn.Module):
    def __init__(self, num_seg_classes, num_det_classes, num_anchors=2):
        """
        Args:
            num_seg_classes: Number of classes for semantic segmentation.
            num_det_classes: Number of classes for detection.
            num_anchors: Number of bounding boxes (anchors) predicted per grid cell.
        """
        super(YOLO_UNet, self).__init__()

        self.num_anchors = num_anchors

        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        self.initial = nn.Sequential(
            backbone.conv1, 
            backbone.bn1,
            backbone.relu,
            backbone.maxpool 
        )
        self.encoder1 = backbone.layer1  # 64 channels.
        self.encoder2 = backbone.layer2  # 128 channels.
        self.encoder3 = backbone.layer3  # 256 channels.
        self.encoder4 = backbone.layer4  # 512 channels.

        # -----------------------
        # Segmentation Head (U-Net Decoder)
        # -----------------------
        self.up1 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.up2 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.up3 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.segmentation_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)

        # -----------------------
        # Detection Head (YOLO Style)
        # -----------------------
        self.det_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_det_classes + num_anchors * 5, kernel_size=1)
        )
        self.det_pool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, x):
        # -----------------------
        # Backbone: Encoder (with skip connections)
        # -----------------------
        x0 = self.initial(x)      # Initial feature map.
        x1 = self.encoder1(x0)    # Low-level features (64 channels).
        x2 = self.encoder2(x1)    # (128 channels).
        x3 = self.encoder3(x2)    # (256 channels).
        x4 = self.encoder4(x3)    # (512 channels).

        # -----------------------
        # Detection Branch (YOLO)
        # -----------------------
        det_out = self.det_head(x4)
        det_out = self.det_pool(det_out)
        det_out = det_out.permute(0, 2, 3, 1)

        if not self.training:
            num_det_classes = det_out.shape[-1] - self.num_anchors * 5
            cls_preds = det_out[..., :num_det_classes]   # (N, 7, 7, num_det_classes)
            bbox_preds = det_out[..., num_det_classes:]    # (N, 7, 7, num_anchors*5)
            bbox_preds = bbox_preds.view(det_out.shape[0], 7, 7, self.num_anchors, 5)
            bbox_preds[..., 0:2] = torch.sigmoid(bbox_preds[..., 0:2])
            bbox_preds[..., 4] = torch.sigmoid(bbox_preds[..., 4])
            bbox_preds = bbox_preds.view(det_out.shape[0], 7, 7, -1)
            det_out = torch.cat([cls_preds, bbox_preds], dim=-1)

        # -----------------------
        # Segmentation Branch (U-Net Decoder)
        # -----------------------
        d1 = self.up1(x4, x3)  # Upsample and merge with encoder3.
        d2 = self.up2(d1, x2)  # Upsample and merge with encoder2.
        d3 = self.up3(d2, x1)  # Upsample and merge with encoder1.
        d4 = self.up4(d3)      # Final upsampling.
        seg_out = self.segmentation_head(d4)  
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return seg_out, det_out
