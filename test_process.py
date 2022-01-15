import gym
import numpy as np
import torch
import Config
import NN
from copy import deepcopy

class TestProcess:

    def __init__(self, state_size, action_size, writer):
        self.env = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)

    def test(self, trained_policy_nn, n_step, writer, env):
        self.env = deepcopy(env)
        self.policy_nn.load_state_dict(trained_policy_nn.state_dict())
        state = self.env.reset()
        ep_reward = 0
        ep_reward_mean = [0]*100
        print("Testing...")
        print("Episodes done [", end="")
        for n_episode in range(Config.NUMBER_OF_EPISODES):
            print('.', end="")
            while True:
                #if n_episode % 25 == 0:
                #    self.env.render()
                actions, _, _ = self.policy_nn(torch.tensor(state, dtype=torch.float, device=self.device))
                new_state, reward, done, _ = self.env.step(actions.cpu().detach().numpy())
                ep_reward = ep_reward + reward * np.sqrt(self.env.return_rms.var + self.env.epsilon) if Config.ENV_SCALE_CROP else ep_reward + reward
                state = new_state
                if done:
                    state = self.env.reset()
                    ep_reward_mean[n_episode % 100] = ep_reward
                    ep_reward = 0
                    break
        print("]")
        print("  Mean 100 test reward: " + str(np.round(np.mean(self.env.return_queue), 2)))
        if writer is not None:
            writer.add_scalar('testing 100 reward', np.mean(self.env.return_queue), n_step)
        print("Done testing!")


