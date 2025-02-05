import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, point: bool=False):
        super().__init__()

        # Decide if the convolution is done ewther in a standard way or point-wise
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups) if not point else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        
        # Normalization and activation
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, point_groups: int, bnorm: bool=False):
        super().__init__()
        
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # Decide whether including a batch normalization layer or not
        if bnorm:
            self.__setattr__('depth_bn', torch.nn.BatchNorm2d(in_channels))
        
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=point_groups)
        # Decide whether including a batch normalization layer or not
        if bnorm:
            self.__setattr__('point_bn', torch.nn.BatchNorm2d(out_channels))

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        if hasattr(self, 'depth_bn'):
            x = self.depth_bn(x)

        x = self.pointwise(x)
        if hasattr(self, 'point_bn'):
            x = self.point_bn(x)

        x = self.activation(x)
        return x