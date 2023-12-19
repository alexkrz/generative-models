from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.001,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def loss_function(self, x, x_hat, mean, log_var):
        # Loss function from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=1), dim=0
        )

        return recons_loss + kld_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        imgs, _ = batch
        N, C, H, W = imgs.size()
        x = imgs.reshape(N, -1)

        x_hat, mean, log_var = self(x)
        loss = self.loss_function(x, x_hat, mean, log_var)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
