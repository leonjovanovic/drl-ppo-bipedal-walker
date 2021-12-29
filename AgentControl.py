import Config
import NN
import torch

class AgentControl:

    def __init__(self, state_size, action_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn_old = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)
        self.policy_nn_new = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)
        self.sync_nns()
        self.critic_nn = NN.CriticNN(input_shape=state_size).to(self.device)

    def get_action(self, state):
        # Using old policy to collect data
        actions = self.policy_nn_old(torch.tensor(state, dtype=torch.float, device=self.device))
        return actions

    def advantage(self, ep_state, ep_new_state, ep_reward):
        # [25]
        v_s = self.critic_nn(torch.tensor(ep_state, dtype=torch.float, device=self.device)).squeeze(-1)
        # number [1]
        v_ns = self.critic_nn(torch.tensor(ep_new_state[-1], dtype=torch.float, device=self.device)).squeeze(-1)
        gt = v_ns.item()
        gts = [0]*len(ep_state)
        for i in range(len(ep_state) - 1, -1, -1):
            gts[i] = ep_reward[i] + Config.GAMMA * gt
            gt = gts[i]
        gts_tensor = torch.tensor(gts, dtype=torch.float, device=self.device)
        return (gts_tensor - v_s).detach()


    def ratio(self, ep_state, ep_action):
        new_action = self.policy_nn_new(torch.tensor(ep_state, dtype=torch.float, device=self.device))
        action_tensor = torch.tensor(ep_action, dtype=torch.float, device=self.device)
        # TODO
        # TODO
        # TODO

    def sync_nns(self):
        self.policy_nn_new.load_state_dict(self.policy_nn_old.state_dict())

