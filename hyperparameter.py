import torch

print_step = 100
render_step = 500
n_training_episodes = 5000
reward_scale = 0.01
gamma = 0.99
lr = 0.0008
env_id = 'LunarLander-v2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model_save_name = 'Lunarlander_reinforce' # uncomment for reinforce
model_save_name = 'Lunarlander_reinforce_baseline' # uncomment for reinforce with baseline
frame_render = 5