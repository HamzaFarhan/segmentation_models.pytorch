import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, SCSEModule
from pixel_shuffle import PixelShuffle_ICNR
from ..base.model import Model

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_channels=[], use_batchnorm=True,
                 attention_type=None, upsample='transpose', shuffle_blur=True):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)
        upsample_dict = {'shuffle': PixelShuffle_ICNR(upsample_channels, upsample_channels, scale=2, blur=shuffle_blur),
                         'transpose': nn.ConvTranspose2d(upsample_channels, upsample_channels, kernel_size=2, stride=2)}
        self.up = upsample_dict[upsample]
        # self.shuffle = PixelShuffle_ICNR(upsample_channels, upsample_channels, scale=2, blur=shuffle_blur)


        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x

        x = self.up(x)

        # x = F.pixel_shuffle(x,2)
        # x = self.conv1(x)        

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x

        # x, skip = x
        # print(f'before: {x.shape}')
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        # if skip is not None:
        #     # print(f'skip: {skip.shape}')
        #     x = torch.cat([x, skip], dim=1)
        #     x = self.attention1(x)
        # # print(f'after concat: {x.shape}')
        # x = self.block(x)
        # x = self.attention2(x)
        # # print(f'after block: {x.shape}')
        # return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None,
            upsample='transpose',
            shuffle_blur=True
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels
        upsample_channels = encoder_channels[:1]+decoder_channels[:-1]
        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], upsample_channels[0],
                                   upsample=upsample, shuffle_blur=shuffle_blur,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], upsample_channels[1],
                                   upsample=upsample, shuffle_blur=shuffle_blur,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], upsample_channels[2],
                                   upsample=upsample, shuffle_blur=shuffle_blur,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], upsample_channels[3],
                                   upsample=upsample, shuffle_blur=shuffle_blur,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], upsample_channels[4],
                                   upsample=upsample, shuffle_blur=shuffle_blur,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
