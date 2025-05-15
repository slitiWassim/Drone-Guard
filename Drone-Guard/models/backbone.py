import logging
import sys
from collections import OrderedDict
from functools import partial
import torch.nn as nn
import torch
Norm2d = nn.BatchNorm2d
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.hub import load_state_dict_from_url
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import EfficientX3d



model_efficient_x3d_xs = EfficientX3d(expansion='XS', head_act='identity')
checkpoint_path = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth'
checkpoint = load_state_dict_from_url(checkpoint_path)
model_efficient_x3d_xs.load_state_dict(checkpoint)
s1=model_efficient_x3d_xs.s1
s2=model_efficient_x3d_xs.s2
s3=model_efficient_x3d_xs.s3
s4=model_efficient_x3d_xs.s4


def bnrelu(channels):
    """
    Single Layer BN and Relui
    """
    return nn.Sequential(Norm2d(channels),
                         nn.ReLU(inplace=True))


class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling over the input's spatial dimensions
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):
    """
    Identity Residual Block for WideResnet
    """
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None,
                 dist_bn=False
                 ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn


        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels,
                                    channels[0],
                                    3,
                                    stride=stride,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           padding=0,
                           bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    padding=dilation, bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class Efficientnet(nn.Module):
    """
    This is Efficientnet
    """
    def __init__(self, pretrained=True):
        super(Efficientnet, self).__init__()
        self.model_f =efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        
        self.mod10=self.model_f[0]
        self.mod11=self.model_f[1]
        self.mod12=self.model_f[2]
        self.mod13=self.model_f[3]
        self.mod14=self.model_f[4]
        self.mod15=self.model_f[5]
        self.mod16=self.model_f[6]
        self.mod17=self.model_f[7]

    def forward(self, x):
        x=self.mod11(self.mod10(x))
        s2_features=x
        
        x=self.mod12(x)
        s4_features=x
        
        x=self.mod13(x)
        s3_features=x
        
        x=self.mod14(x)
       
        x=self.mod16(self.mod15(x))
        #x=self.mod17(x)
        
        return s2_features, s4_features,s3_features, x


class Efficientnet_1024(nn.Module):
    """
    This is Efficientnet_1024
    """
    def __init__(self, pretrained=True):
        super(Efficientnet_1024, self).__init__()
        self.model_f =efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        
        self.mod10=self.model_f[0]
        self.mod11=self.model_f[1]
        self.mod12=self.model_f[2]
        self.mod13=self.model_f[3]
        self.mod14=self.model_f[4]
        self.mod15=self.model_f[5]
        self.mod16=self.model_f[6]
        self.mod17=self.model_f[7]

    def forward(self, x):
        x=self.mod11(self.mod10(x))
        s2_features=x
        x=self.mod12(x)
        s4_features=x
        x=self.mod14(self.mod13(x))
        s3_features=x
        x=self.mod16(self.mod15(x))
        x=self.mod17(x)
        return s2_features, s4_features,s3_features, x        





class Efficientnet_X3D(nn.Module):
    """
    This is Efficientnet_X3D
    """
    def __init__(self, pretrained=True):
        super(Efficientnet_X3D, self).__init__()
        self.mod1=s1
        self.mod2=s2
        self.mod3=s3

 

    def forward(self, x):
        x1=self.mod1(x)
        x2=self.mod2(x1)
        x3=self.mod3(x2)
        return x1 , x2 , x3    


   
