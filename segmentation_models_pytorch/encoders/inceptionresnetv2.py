import torch.nn as nn
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2
from pretrainedmodels.models.inceptionresnetv2 import pretrained_settings

from .base import EncoderMixin


class InceptionResNetV2Encoder(InceptionResNetV2, EncoderMixin):

    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = depth

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def forward(self, x):

        modules = [
            nn.Identity(),
            nn.Sequential(self.conv2d_1a, self.conv2d_2a, self.conv2d_2b),
            nn.Sequential(self.maxpool_3a, self.conv2d_3b, self.conv2d_4a),
            nn.Sequential(self.maxpool_5a, self.mixed_5b, self.repeat),
            nn.Sequential(self.mixed_6a, self.repeat_1),
            nn.Sequential(self.mixed_7a, self.repeat_2, self.block8, self.conv2d_7b),
        ]

        features = []
        for i in range(self._depth + 1):
            x = modules[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


inceptionresnetv2_encoders = {
    'inceptionresnetv2': {
        'encoder': InceptionResNetV2Encoder,
        'pretrained_settings': pretrained_settings['inceptionresnetv2'],
        'out_shapes': (1536, 1088, 320, 192, 64),
        'out_channels': (3, 64, 192, 320, 1088, 1536),
        'params': {
            'num_classes': 1000,
        }

    }
}
