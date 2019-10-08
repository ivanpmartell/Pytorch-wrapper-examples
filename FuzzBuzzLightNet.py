import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.FuzzBuzzDataset import FuzzBuzzDataset

import pytorch_lightning as pl


class FuzzBuzzModel(pl.LightningModule):

    def __init__(self, input_size, hidden_size, output_size):
        """
                2 layer network for prediction fiz or buz
                :param int input_size: size of the input layer
                :param int hidden_size: size of the hidden linear layer
                :param int output_size: size of the output linear layer
                """
        super(FuzzBuzzModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.hidden(x)
        activated = torch.sigmoid(hidden)
        out = self.out(activated)
        return out

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.01)]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = FuzzBuzzDataset(10)
        return DataLoader(dataset, pin_memory=True, batch_size=64, shuffle=True, num_workers=0)

    @pl.data_loader
    def val_dataloader(self):
        dataset = FuzzBuzzDataset(10)
        return DataLoader(dataset, pin_memory=True, batch_size=128, shuffle=True, num_workers=0)

    @pl.data_loader
    def test_dataloader(self):
        dataset = FuzzBuzzDataset(10)
        return DataLoader(dataset, pin_memory=True, batch_size=128, shuffle=True, num_workers=0)