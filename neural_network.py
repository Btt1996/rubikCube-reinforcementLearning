import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_shape, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
