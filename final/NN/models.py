import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

cfg = Config()


class hahow_net(nn.Module):
    def __init__(self, cfg):
        super(hahow_net, self).__init__()

        self.layer1 = self._make_layer(cfg.in_features, cfg.in_features)
        self.layer2 = self._make_layer(cfg.in_features, cfg.in_features)
        self.layer3 = self._make_layer(cfg.in_features, cfg.in_features)
        self.layer4 = self._make_layer(cfg.in_features, cfg.course_num)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(p=0.2, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        )

    def forward(self, x):
        out = self.layer1(x)
        if cfg.challenge == 'unseen':
            out = self.layer2(out)
            out = self.layer3(out)
        out = self.layer4(out)

        return out
