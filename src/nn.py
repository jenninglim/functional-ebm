import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class Generator(torch.nn.Module):
    def __init__(self, latent_dim, weight_dim, y_dim, n_hidden):
        super(Generator, self).__init__()
        # branch
        # self.encoder = DomainEncoder(weights_dim, 100)
        self.fc1 = (nn.Linear(latent_dim, n_hidden))
        self.act1 = nn.ReLU()
        self.fc2 = (nn.Linear(n_hidden, n_hidden))
        self.act2 = nn.ReLU()
        self.fc3 = (nn.Linear(n_hidden, n_hidden))
        self.act3 = nn.ReLU()
        self.fc4 = (nn.Linear(n_hidden, (weight_dim * y_dim)))
        self.weight_dim = weight_dim
        self.y_dim = y_dim

    def forward(self, weights):
        # weights = self.encoder(weights)
        weights = self.act1(self.fc1(weights))
        weights = self.act2(self.fc2(weights)) + weights
        weights = self.act3(self.fc3(weights)) + weights
        weights = self.fc4(weights).reshape(*weights.shape[:-1],
                                            self.weight_dim,
                                            self.y_dim)
        return weights


class Energy(torch.nn.Module):
    def __init__(self, in_dim, n_hidden=512):
        super(Energy, self).__init__()
        # branch
        self.fc1 = nn.Linear(in_dim, n_hidden)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.act3 = nn.SiLU()
        self.fc4 = nn.Linear(n_hidden, 1)

    def forward(self, weights):
        weights = self.act1(self.fc1(weights))
        weights = self.act2(self.fc2(weights))
        weights = self.act3(self.fc3(weights))
        weights = self.fc4(weights)
        return weights
