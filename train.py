import torch
import gym
import torch
import imageio
import numpy as np
import matplotlib as plt
import torch.optim as optim
from model import Policy
from collections import deque
from hyperparameter import(
    print_step,
    render_step,
    n_evaluation_episodes,
    n_training_episodes,
    max_t,
    max_t_sim,
    gamma,
    lr,
    env_id,
    device,
    t_name,
    e_name,
    model_name
)

def reinforce(name, policy, optimizer, n_episodes, max_t, max_t_sim, gamma, print_step, render_step, env):
    scores_deque = deque(maxlen = 100)
    scores = []

    for e in range(1, n_episodes + 1):
        log_probs = []
        rewards = []
        state, _ = env.reset()
        
        for t in range(max_t):
            action, log_prob = policy.act(state)
            log_probs.append(log_prob)
            state, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
            if terminated:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen = max_t)
        n_steps = len(rewards)
        pw = 0
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma ** pw * disc_return_t + rewards[t])
            pw = pw + 1
        
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for lp, dr in zip(log_probs, returns):
            policy_loss.append(-lp * dr)
        
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.requires_grad = True
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if e % print_step == 0:
            print(f'Episode {e}\tAverage Score: {np.mean(scores_deque)}')

        if e % render_step == 0 or e == 1:
            frames = []
            state, _ = env.reset()
            for _ in range(max_t_sim):
                action, _ = policy.act(state)
                state, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                frames.append(frame)
                imageio.mimsave(f'simulations/{name}_simulation_epoch_{e}.gif', frames)

                if terminated or truncated:
                    print(f'video recorded for epoch {e}')
                    break

    return scores

def evaluate_agent(name, env, max_steps, n_eval_episodes, policy):
    policy.eval()
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        total_rewards_ep = 0
        frames = []
        for _ in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, terminated, _, _ = env.step(action)
            total_rewards_ep += reward

            frame = env.render()
            frames.append(frame)
            imageio.mimsave(f'simulations/{name}_simulation_epoch_{episode}.gif', frames)

            if terminated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

if __name__ == '__main__':
    env = gym.make(env_id, render_mode = 'rgb_array')
    eval_env = gym.make(env_id, render_mode = 'rgb_array')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    policy = Policy(state_size, action_size).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr = lr)

    scores = reinforce(t_name, policy, optimizer, n_training_episodes, max_t, max_t_sim, gamma, print_step, render_step, env)

    mr, stdr = evaluate_agent(e_name, eval_env, max_t_sim, n_evaluation_episodes, policy)     

    torch.save({'model_state_dict': policy.state_dict()}, f'checkpoints/{model_name}_policy_checkpoint.pth')

    print('evaluation:')
    print(f'mean reward {mr}\tstd_reward: {stdr}')