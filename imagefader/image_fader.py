import torch
from torch import nn

from imagevae.mnist_vae import MnistVAE
from imagevae.dsprites_vae import DspritesVAE
from imagevae.image_vae_trainer import MNIST_REG_TYPES, DSPRITES_REG_TYPE


class ImageFaderDiscriminator(nn.Module):
    def __init__(self, num_attributes):
        super().__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(16, 64),
            nn.Dropout(0.5),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.SELU(),
            nn.Linear(32, num_attributes),
        )

    def forward(self, z):
        preds = self.fc_seq(z)
        return torch.sigmoid(preds)


class MnistFaderNetwork(MnistVAE):
    """
    Class defining a fader network for MNIST images
    """
    def __init__(self):
        super(MnistFaderNetwork, self).__init__()
        self.num_attributes = len(MNIST_REG_TYPES) - 1
        self.dec_lin = nn.Sequential(
            nn.Linear(self.z_dim + self.num_attributes, 256),
            nn.SELU(),
            nn.Linear(256, 2888),
            nn.SELU()
        )

    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'MnistFader' + self.trainer_config

    def encode(self, x):
        hidden = self.enc_conv(x)
        hidden = hidden.view(x.size(0), -1)
        hidden = self.enc_lin(hidden)
        z = self.enc_mean(hidden)
        return z

    def forward(self, x, labels):
        """
        Implements the forward pass of the Fader AE
        :param x: minist image input
            (batch_size, 28, 28)

        """
        # compute output of the encoding layer
        z = self.encode(x)

        # compute output of decoding layer
        dec_in = torch.cat((z, labels), 1)
        output = self.decode(dec_in)

        return output, z


class DspritesFaderNetwork(DspritesVAE):
    """
    Class defining a fader network for Dsprites datset
    """
    def __init__(self):
        super().__init__()
        self.num_attributes = len(DSPRITES_REG_TYPE) - 1
        self.dec_lin = nn.Sequential(
            nn.Linear(self.z_dim + self.num_attributes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )

    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'DspritesFader' + self.trainer_config

    def encode(self, x):
        hidden = self.enc_conv(x)
        hidden = hidden.view(x.size(0), -1)
        hidden = self.enc_lin(hidden)
        z = self.enc_mean(hidden)
        return z

    def forward(self, x, labels):
        """
        Implements the forward pass of the Fader AE
        :param x: minist image input
            (batch_size, 28, 28)

        """
        # compute output of the encoding layer
        z = self.encode(x)

        # compute output of decoding layer
        dec_in = torch.cat((z, labels), 1)
        output = self.decode(dec_in)

        return output, z
