import torch
from torch import nn
from model import Model


class Network(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Network, self).__init__()

        # encoder
        model = Model().cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_path))

        self.layer = model.module.f
        # classifier
        self.fully_connected = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.layer(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fully_connected(feature)
        return out
