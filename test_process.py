import gym
import numpy as np
import torch
import Config
import NN
from copy import deepcopy
import sys

class TestProcess:

    def __init__(self, state_size, action_size, writer):
        self.env = gym.make(Config.ENV_NAME)
        if Config.ENV_SCALE_CROP:
            self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
            self.env = gym.wrappers.ClipAction(self.env)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)

    def test(self, trained_policy_nn, n_step, writer, env):
        #self.env = deepcopy(env)
        self.policy_nn.load_state_dict(trained_policy_nn.state_dict())
        self.env = gym.make(Config.ENV_NAME)
        if Config.ENV_SCALE_CROP:
            self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
            self.env = gym.wrappers.ClipAction(self.env)
        state = self.env.reset()
        state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
        state = np.clip(state, -10, 10)
        print("Testing...")
        print("Episodes done [", end="")
        for n_episode in range(Config.NUMBER_OF_EPISODES):
            print('.', end="")
            while True:
                #if n_episode % 25 == 0:
                #    self.env.render()
                actions, _, _ = self.policy_nn(torch.tensor(state, dtype=torch.float, device=self.device))
                new_state, reward, done, _ = self.env.step(actions.cpu().detach().numpy())
                state = new_state
                state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
                state = np.clip(state, -10, 10)
                if done:
                    state = self.env.reset()
                    break
        print("]")
        print("  Mean 100 test reward: " + str(np.round(np.mean(self.env.return_queue), 2)))
        print(self.env.return_queue)
        if writer is not None:
            writer.add_scalar('testing 100 reward', np.mean(self.env.return_queue), n_step)
        print("Done testing!")
        if np.mean(self.env.return_queue) >= 300:
            print("Goal reached! Mean reward over 100 episodes is " + str(np.mean(self.env.return_queue)))
            model_scripted = torch.jit.script(self.policy_nn) # Export to TorchScript
            model_scripted.save('model_scripted_' + Config.date_time+'.pt') # Save
            self.env = gym.wrappers.RecordVideo(env, "bestRecordings", name_prefix="rl-video" + Config.date_time,)
            state = self.env.reset()
            state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
            while True:
                actions, _, _ = self.policy_nn(torch.tensor(state, dtype=torch.float, device=self.device))
                new_state, reward, done, _ = self.env.step(actions.cpu().detach().numpy())
                state = new_state
                state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
                state = np.clip(state, -10, 10)
                if done:
                    break
            return True
        return False


