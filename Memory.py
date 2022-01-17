import Config
import torch

class Memory:

    def __init__(self, state_size, action_size, batch_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.states = torch.zeros(batch_size, state_size).to(self.device)
        self.actions = torch.zeros(batch_size, action_size).to(self.device)
        self.action_logprobs = torch.zeros(batch_size, action_size).to(self.device)
        self.new_states = torch.zeros(batch_size, state_size).to(self.device)
        self.rewards = torch.zeros(batch_size).to(self.device)
        self.dones = torch.zeros(batch_size).to(self.device)
        self.advantages = torch.zeros(batch_size + 1).to(self.device)
        self.gt = torch.zeros(batch_size + 1).to(self.device)
        self.old_value_state = torch.zeros(batch_size).to(self.device)

    def add(self, state, action, actions_logprob, new_state, reward, done, n_batch_step):
        self.states[n_batch_step] = torch.Tensor(state).to(self.device)
        self.actions[n_batch_step] = torch.Tensor(action).to(self.device)
        self.action_logprobs[n_batch_step] = actions_logprob.detach()
        self.new_states[n_batch_step] = torch.Tensor(new_state).to(self.device)
        self.rewards[n_batch_step] = torch.Tensor((reward, )).squeeze(-1).to(self.device)
        self.dones[n_batch_step] = torch.Tensor((int(done is True), )).squeeze(-1).to(self.device)

    def set_old_value_state(self, old_v_s):
        self.old_value_state = old_v_s

    def calculate_advantage(self, next_value, values):
        gt = next_value
        for i in reversed(range(self.batch_size)):
            gt = self.rewards[i] + Config.GAMMA * gt * (1 - self.dones[i])
            self.gt[i] = gt
            self.advantages[i] = gt - values[i]

    def calculate_gae_advantage(self, values, next_values):
        self.gt[self.batch_size] = next_values[-1]
        for i in reversed(range(self.batch_size)):
            delta = self.rewards[i] + Config.GAMMA * next_values[i] * (1 - self.dones[i]) - values[i]
            self.advantages[i] = delta + Config.LAMBDA * Config.GAMMA * self.advantages[i+1] * (1 - self.dones[i])
            # For critic
            self.gt[i] = self.rewards[i] + Config.GAMMA * self.gt[i+1] * (1 - self.dones[i])

