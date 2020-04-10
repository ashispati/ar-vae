import torch
from torch import nn, distributions

from utils.model import Model


class MnistVAE(Model):
    """
    Class defining a variational auto-encoder (VAE) for MNIST images
    """
    def __init__(self):
        super(MnistVAE, self).__init__()
        self.input_size = 784
        self.z_dim = 16
        self.inter_dim = 19
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 8, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
        )
        self.enc_lin = nn.Sequential(
            nn.Linear(2888, 256),
            nn.SELU()
        )
        self.enc_mean = nn.Linear(256, self.z_dim)
        self.enc_log_std = nn.Linear(256, self.z_dim)
        self.dec_lin = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.SELU(),
            nn.Linear(256, 2888),
            nn.SELU()
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 64, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 64, 4, 1),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 1, 4, 1),
        )
        self.xavier_initialization()

        self.update_filepath()

    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'MnistVAE' + self.trainer_config

    def encode(self, x):
        hidden = self.enc_conv(x)
        hidden = hidden.view(x.size(0), -1)
        hidden = self.enc_lin(hidden)
        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution

    def decode(self, z):
        hidden = self.dec_lin(z)
        hidden = hidden.view(z.size(0), -1, self.inter_dim, self.inter_dim)
        hidden = self.dec_conv(hidden)
        return hidden

    def reparametrize(self, z_dist):
        """
        Implements the reparametrization trick for VAE
        """
        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def forward(self, x):
        """
        Implements the forward pass of the VAE
        :param x: minist image input
            (batch_size, 28, 28)

        """
        # compute distribution using encoder
        z_dist = self.encode(x)

        # reparametrize
        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        # compute output of decoding layer
        output = self.decode(z_tilde).view(x.size())

        return output, z_dist, prior_dist, z_tilde, z_prior
