import torch
from torchvision.models.resnet import ResNet, BasicBlock

from utils.model import Model


class MnistResNet(ResNet, Model):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avgpool = torch.nn.AvgPool2d(1, stride=1)
        self.trainer_config = ''
        self.update_filepath()

    def __repr__(self):
        return 'MnistRESNET'

    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)