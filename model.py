import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from hyperparameter import device

# class Policy(nn.Module):
    
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         super(Policy, self).__init__()
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, action_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         action_probs = F.softmax(self.fc2(x), dim = -1)
#         return action_probs
    
#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         action_probs = self.forward(state)
#         highest_prob_action = np.random.choice(self.action_size, p = np.squeeze(action_probs.detach().cpu().numpy()))
#         log_prob = torch.log(action_probs.squeeze(0)[highest_prob_action])
#         return highest_prob_action, log_prob

class Policy(nn.Module):
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5_mean = nn.Linear(128, action_size) 
        self.fc5_stddev = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mean = torch.tanh(self.fc5_mean(x))
        stddev = F.softplus(self.fc5_stddev(x))
        return mean, stddev
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, stddev = self.forward(state)
        # Sample an action from a Gaussian distribution for each dimension
        action = torch.normal(mean, stddev)
        action_dist = Normal(mean, stddev)
        log_prob = action_dist.log_prob(action)

        return action.squeeze().detach().cpu().numpy(), log_prob.sum().item()