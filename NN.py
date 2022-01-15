import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import Config

torch.manual_seed(Config.SEED)
class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_shape)
        )
        self.actions_logstd = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x, actions=None):
        actions_mean = self.actions_mean(x)
        actions_logstd = self.actions_logstd
        actions_std = torch.exp(actions_logstd)
        prob = Normal(actions_mean, actions_std)
        if actions is None:
            actions = prob.sample()
        return actions, prob.log_prob(actions), torch.sum(prob.entropy(), dim=-1)


class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        #self.model.double()????

    def forward(self, x):
        return self.model(x)
