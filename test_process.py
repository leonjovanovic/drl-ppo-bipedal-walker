import gym
import numpy as np
import torch
import Config
import NN

class TestProcess:

    def __init__(self, state_size, action_size, writer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)

    def test(self, trained_policy_nn, n_step, writer, env):
        self.policy_nn.load_state_dict(trained_policy_nn.state_dict())
        state = env.reset()
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
                new_state, reward, done, _ = env.step(actions.cpu().detach().numpy())
                ep_reward += reward
                state = new_state
                if done:
                    state = env.reset()
                    ep_reward_mean[n_episode % 100] = ep_reward
                    ep_reward = 0
                    break
        print("]")
        print("  Mean 100 test reward: " + str(np.round(np.mean(ep_reward_mean), 2)))
        if writer is not None:
            writer.add_scalar('testing 100 reward', np.mean(ep_reward_mean), n_step)
        print("Done testing!")
        return env.reset()


