from torchvision.models.resnet import resnet18 as tv_resnet18
import torch
from functools import partial


def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # x = self.fc(x) get feature
    return x


def resnet18(pretrained=False):
    model = tv_resnet18(pretrained=pretrained)
    model.__call__ = _forward_impl
    return model
