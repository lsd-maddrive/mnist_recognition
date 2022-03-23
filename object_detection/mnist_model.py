import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size1=200,
        hidden_size2=150,
        hidden_size3=100,
        hidden_size=80,
        output=10,
    ):
        super().__init__()
        self.f_connected1 = nn.Linear(input_size, hidden_size1)
        self.f_connected2 = nn.Linear(hidden_size1, hidden_size2)
        self.f_connected3 = nn.Linear(hidden_size2, hidden_size3)
        self.f_connected4 = nn.Linear(hidden_size3, hidden_size)
        self.out_connected = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = F.relu(self.f_connected1(x))
        out = F.relu(self.f_connected2(out))
        out = F.relu(self.f_connected3(out))
        out = F.relu(self.f_connected4(out))
        out = self.out_connected(out)
        return out
