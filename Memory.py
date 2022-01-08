import Config
import torch
import numpy as np

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
        self.advantages = torch.zeros(batch_size).to(self.device)
        self.gt = torch.zeros(batch_size).to(self.device)

    def add(self, state, action, actions_logprob, new_state, reward, done, n_batch_step):
        self.states[n_batch_step] = torch.Tensor(state).to(self.device)
        self.actions[n_batch_step] = torch.Tensor(action).to(self.device)
        self.action_logprobs[n_batch_step] = actions_logprob
        self.new_states[n_batch_step] = torch.Tensor(new_state).to(self.device)
        self.rewards[n_batch_step] = torch.Tensor((reward, )).squeeze(-1).to(self.device)
        self.dones[n_batch_step] = torch.Tensor((int(done is True), )).squeeze(-1).to(self.device)

    def calculate_advantage(self, next_value, values):
        #print(next_value)
        gt = next_value
        for i in reversed(range(self.batch_size)):
            gt = self.rewards[i] + Config.GAMMA * gt * (1 - self.dones[i])
            self.gt[i] = gt
            self.advantages[i] = gt - values[i]

