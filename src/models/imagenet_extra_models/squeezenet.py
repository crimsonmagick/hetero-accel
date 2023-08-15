import os
import  logging
import re
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from core.layers import PrunedConv, PrunedLinear


msglogger = logging.getLogger()

# you need to download the models to ~/.torch/models
model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

squeeze1_0_model_name = 'squeezenet1_0-a815701f.pth'
squeeze1_1_model_name = 'squeezenet1_1-f364aa15.pth'
models_dir = os.path.expanduser('~/.torch/models')


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = PrunedConv(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = PrunedConv(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = PrunedConv(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.squeeze_activation(self.squeeze(X))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(X)),
            self.expand3x3_activation(self.expand3x3(X))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                PrunedConv(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                PrunedConv(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        final_conv = PrunedConv(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, PrunedConv):
                if m is final_conv:
                    init.normal_(m.wrapped_module.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.wrapped_module.weight.data)
                if m.wrapped_module.bias is not None:
                    m.wrapped_module.bias.data.zero_()

    def forward(self, x):
        # if not self.training:
        #     x = quantize_stochastic_rounding(x, 4, 12)

        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained=False, num_classes=1000, **kwargs):
    model = SqueezeNet(1.0, num_classes=num_classes)

    if pretrained:
        if num_classes == 200:
            raise ValueError("Tiny-imagenet models do not have preloaded checkpoints")

        state_dict = load_state_dict_from_url(model_urls['squeezenet1_0'], progress=True)
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            new_name = name

            if re.sub('(weight|bias)', r'wrapped_module.\1', name) in model.state_dict().keys():
                new_name = re.sub('(weight|bias)', r'wrapped_module.\1', name)
            new_state_dict[new_name] = param

        # add masks to the new state dict
        for name, param in model.state_dict().items():
            if 'mask' in name and name not in new_state_dict:
                new_state_dict[name] = param

        anomalous_keys = model.load_state_dict(new_state_dict, strict=False)
        if anomalous_keys:
            missing_keys, unexpected_keys = anomalous_keys
            msglogger.debug(f'Missing keys: {missing_keys}')
            msglogger.debug(f'Unexpected keys: {unexpected_keys}')

            if unexpected_keys:
                msglogger.warning(f"Warning: the loaded checkpoint ({model_urls['squeezenet1_0']}) contains {len(unexpected_keys)} "
                                  f"unexpected state keys")
            if missing_keys:
                raise ValueError(f"The loaded checkpoint ({model_urls['squeezenet1_0']}) is missing {len(missing_keys)} state keys")

    # if pretrained:
    #     model.load_state_dict(torch.load(os.path.join(models_dir, squeeze1_0_model_name)))
    return model


def squeezenet1_1(pretrained=False, num_classes=1000, **kwargs):
    model = SqueezeNet(1.1, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, squeeze1_1_model_name)))
    return model

