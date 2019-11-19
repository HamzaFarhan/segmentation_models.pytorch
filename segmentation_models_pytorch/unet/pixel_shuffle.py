import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, SCSEModule
from ..base.model import Model


def relu(inplace=False, leaky=None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)

def conv_layer(ni, nf, ks=3, stride=1, padding=None, bias=None, norm_type='weight',
               init=nn.init.kaiming_normal_, use_activ=True, bn=True, leaky=None):
    if padding is None: padding = (ks-1)//2
    if bias is None: bias = not bn
    conv = nn.Conv2d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding)
    init(conv.weight)
    if   norm_type.lower()=='weight':   conv = nn.utils.weight_norm(conv)
    elif norm_type.lower() == 'spectral': conv = nn.utils.spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm2d)(nf))
    return nn.Sequential(*layers)

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=True, norm_type='weight', leaky:float=None):
        super(PixelShuffle_ICNR, self).__init__()
        if nf is None:
            nf = ni
        self.conv = conv_layer(ni, nf*(scale**2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = relu(True, leaky=leaky)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x