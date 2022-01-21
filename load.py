import gym
import numpy as np
import torch
import Config
import NN

env = gym.make(Config.ENV_NAME)
if Config.ENV_SCALE_CROP:
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
state = env.reset()
obs_rms_mean =
obs_rms_var =
epsilon =
state = (state - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
state = np.clip(state, -10, 10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy_nn = NN.PolicyNN(input_shape=state.shape[0], output_shape=env.action_space.shape[0]).to(device)
policy_nn.load_state_dict(torch.load('models/model20.20.11.31-s478.p'))
for n_episode in range(Config.NUMBER_OF_EPISODES):
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

