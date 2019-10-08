import torch
from torch import nn


class FuzzBuzzModel(nn.Module):

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

    def forward(self, batch):
        """
        forward pass of the network
        :param torch.Tensor batch: data to input to the network
        :return torch.Tensor: output of the neural network
        """
        hidden = self.hidden(batch)
        activated = torch.sigmoid(hidden)
        out = self.out(activated)
        return out
