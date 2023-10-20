import imageio
import torch
import torch.optim as optim
import numpy as np
from model import Policy
from hyperparameter import device, model_save_name, frame_render

class Agent():
    
    def __init__(self, learning_rate, discount, state_size, action_size, hidden_size = 128):
        self.action_size = action_size
        self.policy_network = Policy(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = learning_rate)
        self.discount = discount

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = self.policy_network(state)
            action_probs = action_probs.detach().cpu().squeeze(0).numpy()
            action = np.random.choice(self.action_size, p = action_probs)
            
        return action
    
    def train(self, state_list, action_list, reward_list):
        len_trajectory = len(reward_list)
        reward_array = np.zeros((len_trajectory))

        ret = 0
        for r in range(len_trajectory - 1, -1, -1):
            ret = reward_list[r] + (self.discount * ret)
            reward_array[r] = ret
        
        state_t = torch.FloatTensor(state_list).to(device)
        action_t = torch.LongTensor(action_list).to(device).view(-1,1)
        return_t = torch.FloatTensor(reward_array).to(device).view(-1,1)

        selected_action_prob = self.policy_network(state_t).gather(1, action_t)
        
        loss = torch.mean(-torch.log(selected_action_prob) * return_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

        return loss.detach().cpu().numpy()
    
    def save_model(self):
        torch.save({'model_state_dict': self.policy_network.state_dict()}, f'checkpoints/{model_save_name}.pth')
    
    def render(self, eps, eval_env):
        frames = []
        total_reward = 0
        step = 1
        state, _ = eval_env.reset()
        with torch.no_grad():
            while True:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                step += 1
                if step % frame_render == 0:
                    frame = eval_env.render()
                    frames.append(frame)

                done = terminated or truncated

                if total_reward <= -250:
                    done = True

                if done:
                    break

                state = next_state

        imageio.mimsave(f'simulations/{model_save_name}_{eps}.gif', frames)
