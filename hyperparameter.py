import torch

print_step = 100
render_step = 250
n_training_episodes = 1000
n_evaluation_episodes = 3
max_t = 10000
max_t_sim = 250
gamma = 0.99
lr = 0.001
env_id = 'Ant-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_name = 'train'
e_name = 'eval'
model_name = 'Ant'