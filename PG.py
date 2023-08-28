import numpy as np
from collections import deque
import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = 'CartPole-v1'
env = gym.make(env_id)
eval_env = gym.make(env_id)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class Policy(nn.Module):
    
    def __init__(self, state_size, action_size, hiddenLayer_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hiddenLayer_size)
        self.fc2 = nn.Linear(hiddenLayer_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(policy, optimizer, n_episodes, max_t, gamma, print_step):
    scores_deque = deque(maxlen = 100)
    scores = []

    for e in range(1, n_episodes + 1):
        log_probs = []
        rewards = []
        state = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen = max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])
        
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for lp, dr in zip(log_probs, returns):
            policy_loss.append(-lp * dr)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_step == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(e, np.mean(scores_deque)))

    return scores


cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": state_size,
    "action_space": action_size,
}

cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr = cartpole_hyperparameters["lr"])

scores = reinforce(
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

mr, stdr = evaluate_agent(
    eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy
)

print('evaluation:')
print("mean reward {}\tstd_reward: {:.2f}".format(mr, stdr))