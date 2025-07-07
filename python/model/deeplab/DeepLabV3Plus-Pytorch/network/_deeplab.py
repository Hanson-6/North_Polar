import torch
from torch import nn
from torch.nn import functional as F

# 导入 ResNet-50 主干及其预训练权重
from torchvision.models import resnet50, ResNet50_Weights
# 用于从主干网络中提取不同层级的特征
from torchvision.models._utils import IntermediateLayerGetter

__all__ = ["DeepLabV3"]

class AtrousSeparableConvolution(nn.Module):
    """带空洞率的深度可分离卷积
    将大核卷积分解为 depthwise + pointwise，加速且减少参数"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1,
                bias=bias
            )
        )
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    """ASPP 中的普通分支：3x3 空洞卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    """ASPP 中的图像级池化分支：全局平均池化 + 1x1 卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling 模块，聚合多尺度上下文"""
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = 256
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            ASPPConv(in_channels, out_channels, atrous_rates[0]),
            ASPPConv(in_channels, out_channels, atrous_rates[1]),
            ASPPConv(in_channels, out_channels, atrous_rates[2]),
            ASPPPooling(in_channels, out_channels)
        ]
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3(nn.Module):
    """DeepLabV3+ 模型：ResNet-50 主干 + ASPP + 解码头 + 上采样"""
    def __init__(self, num_classes=2, backbone_pretrained=True,
                 replace_stride_with_dilation=[False, True, True]):
        super().__init__()
        # Backbone
        weights = ResNet50_Weights.DEFAULT if backbone_pretrained else None
        backbone_full = resnet50(weights=weights,
                                 replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low_level', 'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone_full, return_layers=return_layers)
        # Decoder head
        self.project_low = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(in_channels=2048, atrous_rates=[12,24,36])
        self.classifier = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        low  = self.project_low(feats['low_level'])
        high = self.aspp(feats['out'])
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        y    = self.classifier(torch.cat([low, high], dim=1))
        return F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def convert_to_separable_conv(module: nn.Module) -> nn.Module:
    """
    将所有大核 Conv2d 替换为带空洞率的深度可分离卷积
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size[0] > 1:
            sep = AtrousSeparableConvolution(
                child.in_channels, child.out_channels,
                child.kernel_size, stride=child.stride,
                padding=child.padding, dilation=child.dilation,
                bias=(child.bias is not None)
            )
            setattr(module, name, sep)
        else:
            convert_to_separable_conv(child)
    return module