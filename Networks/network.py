import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, activation_function=torch.tanh, learning_rate=0.9):
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.activation_function = activation_function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, activation_function=F.softplus):
        return F.softplus(self.fc(x))



