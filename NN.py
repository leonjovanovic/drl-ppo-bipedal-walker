import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape),
            nn.Tanh()
        )
        self.actions_logstd = nn.Parameter(torch.zeros(output_shape))
        #self.model.double()????

    def forward(self, x, actions=None):
        #output = F.log_softmax(x, dim=1)
        actions_mean = self.actions_mean(x)
        actions_logstd = self.actions_logstd
        actions_std = torch.exp(actions_logstd)
        prob = Normal(actions_mean, actions_std)
        if actions is None:
            actions = prob.sample()
        return actions, prob.log_prob(actions)


class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #self.model.double()????

    def forward(self, x):
        return self.model(x)
