# Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int, feat_dim: int, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feat_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_dim * 8),
            nn.ReLU(True),
            # state size. ``(feat_dim*8) x 4 x 4``
            nn.ConvTranspose2d(feat_dim * 8, feat_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 4),
            nn.ReLU(True),
            # state size. ``(feat_dim*4) x 8 x 8``
            nn.ConvTranspose2d(feat_dim * 4, feat_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 2),
            nn.ReLU(True),
            # state size. ``(feat_dim*2) x 16 x 16``
            nn.ConvTranspose2d(feat_dim * 2, feat_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
            # state size. ``(feat_dim) x 32 x 32``
            nn.ConvTranspose2d(feat_dim, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(channels) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channels: int, feat_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            # input is ``(channels) x 64 x 64``
            nn.Conv2d(channels, feat_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feat_dim) x 32 x 32``
            nn.Conv2d(feat_dim, feat_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feat_dim*2) x 16 x 16``
            nn.Conv2d(feat_dim * 2, feat_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feat_dim*4) x 8 x 8``
            nn.Conv2d(feat_dim * 4, feat_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feat_dim*8) x 4 x 4``
            nn.Conv2d(feat_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
