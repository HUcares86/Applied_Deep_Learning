import numpy as np
import pandas as pd
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from torch.utils.data import DataLoader
import random

from config import Config
from engine import *
from models import hahow_net

cfg = Config()
if cfg.challenge == 'seen':
    import data_utils
else:
    import data_utils_unseen as data_utils


def main():
    cfg = Config()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set to False guarantee perfect reproducbility, but hurt performance

    # course = pd.read_csv('./data/courses'.csv')
    # user = pd.read_csv('./data/users_preprocessed.csv', usecols=['user_id', 'gender', 'interests'])
    # id2cid = pd.read_csv('./data/id2cid.csv', usecols=['course_id'])
    # id2iid = pd.read_csv('./data/id2iid.csv', usecols=['interests'])
    train = pd.read_csv(cfg.train_file, usecols=['gender', 'interests', 'course_id'])
    print(train.head(5))

    model = hahow_net(cfg)
    model = model.to(cfg.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('###########################################')
    print('batch_size:', cfg.batch_size)
    print('init_lr:', cfg.init_lr)
    print('num_epochs:', cfg.num_epochs)
    print(f"Number of params: {n_parameters}")
    print('###########################################')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=5e-4)

    if cfg.scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if cfg.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3)

    train_data = data_utils.HaHowTrainData(path=cfg.train_file)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    valid_data = data_utils.HaHowValidData(path=cfg.valid_file)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    # accuracy = valid_model(model, valid_loader, criterion, cfg)

    train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg)


if __name__ == "__main__":
    main()
