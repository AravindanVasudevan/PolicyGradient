import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
# from reinforce import Agent # Uncomment for reinforce
from reinforce_baseline import Agent # Uncomment for reinforce with baseline
from hyperparameter import(
    print_step,
    render_step,
    n_training_episodes,
    reward_scale,
    gamma,
    lr,
    env_id
)

if __name__ == '__main__':
    env = gym.make(env_id)
    eval_env = gym.make(env_id, render_mode = 'rgb_array')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(lr, gamma, state_size, action_size)

    total_reward_list = []

    for eps in range(1, n_training_episodes + 1):
        state, _ = env.reset()
        ep_loss = 0
        ep_vn_loss = 0 # Uncomment for reinforce with baseline
        total_reward = 0

        state_list = []
        action_list = []
        reward_list = []

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward * reward_scale)

            done = terminated or truncated

            if total_reward <= -250:
                done = True

            if done:
                # ep_loss = agent.train(state_list, action_list, reward_list) # Uncomment for reinforce
                ep_loss, ep_vn_loss = agent.train(state_list, action_list, reward_list) # Uncomment for reinforce with baseline
                break

            state = next_state
        
        total_reward_list.append(total_reward)
        
        if eps % print_step == 0:
            print(f'Reward obtained at episode {eps} is {total_reward} and loss {ep_loss} and value network loss {ep_vn_loss}')
        
        if eps % render_step == 0:
            print(f'Saving simulation...')
            agent.render(eps, eval_env)
            print(f'Simulation saved!')
    
    episodes = np.arange(1, n_training_episodes + 1)

    plt.plot(episodes, total_reward_list, label = 'Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title('Training Performance')
    plt.show()