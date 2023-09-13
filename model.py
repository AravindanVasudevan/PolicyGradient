import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from hyperparameter import device

class Policy(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, action_size * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_params = self.forward(state).cpu()

        mean, log_std = torch.chunk(action_params, 2, dim = 1)
        action_dist = Normal(mean, torch.exp(log_std))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim = -1)

        return action.numpy()[0], log_prob.item()