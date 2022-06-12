import torch.nn as nn


class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=200, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=200, out_features=150, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=150, out_features=100, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=100, out_features=80, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=80, out_features=10, bias=True),
        )

    def forward(self, x):
        return self.model(x)
