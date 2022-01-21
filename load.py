import gym
import numpy as np
import torch
import Config
import NN
import json

PATH_DATA = 'models/data21.11.43.7.json'
PATH_MODEL = 'models/model20.20.11.31-s478.p'

with open(PATH_DATA, 'r') as f:
    json_load = json.loads(f.read())
obs_rms_mean = np.asarray(json_load["obs_rms_mean"])
obs_rms_var = np.asarray(json_load["obs_rms_var"])
epsilon = json_load["eps"]

env = gym.make(Config.ENV_NAME)
if Config.ENV_SCALE_CROP:
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
state = env.reset()

state = (state - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
state = np.clip(state, -10, 10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy_nn = NN.PolicyNN(input_shape=state.shape[0], output_shape=env.action_space.shape[0]).to(device)
policy_nn.load_state_dict(torch.load(PATH_MODEL))
for n_episode in range(3):
    while True:
        env.render()
        actions, _, _ = policy_nn(torch.tensor(state, dtype=torch.float, device=device))
        new_state, reward, done, _ = env.step(actions.cpu().detach().numpy())
        state = new_state
        state = (state - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
        state = np.clip(state, -10, 10)
        if done:
            state = env.reset()
            break
print(env.return_queue)
