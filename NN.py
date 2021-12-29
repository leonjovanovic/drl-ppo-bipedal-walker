import torch.nn as nn
import torch.nn.functional as F

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_shape),
            nn.Tanh()
        )
        #self.model.double()????

    def forward(self, x):
        #output = F.log_softmax(x, dim=1)
        output = self.model(x)
        return output

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
