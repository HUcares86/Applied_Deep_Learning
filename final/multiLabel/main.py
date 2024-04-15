import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from metric import mapk
import random
import util
import sys
import os
from config import Config

cfg = Config()
path = sys.argv[1]


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(model, epoch, optimizer, training_loader):
    model.train()
    pbar = tqdm(training_loader, desc='ubatch')
    LOSS = []
    for data in pbar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        LOSS.append(loss.item())
        pbar.set_description(f'Epoch:{epoch}, loss={sum(LOSS) / len(LOSS):.4f} ')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation(model, testing_loader, df):

    model.eval()
    val = util.pred_output(model, df, testing_loader, device)
    gt = pd.read_csv(os.path.join(path, 'val_seen_group.csv'))
    gt = gt.fillna("")
    gt = gt['subgroup'].tolist()
    gt = [value.split(' ') for value in gt]
    val = val['subgroup'].tolist()
    val = [value.split(' ') for value in val]
    accuracy = mapk(gt, val, k=50)
    print("acc:", accuracy)
    return val


torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = util.BERTClass()
model.to(device)


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
df = pd.read_csv(os.path.join(path, 'subgroup', "train_preprocess.csv"))
df = df.fillna("")
training_set = util.CustomDataset(df, tokenizer, cfg.MAX_LEN)
df = pd.read_csv(os.path.join(path, 'subgroup', "val_preprocess.csv"))
df = df.fillna("")
testing_set = util.CustomDataset(df, tokenizer, cfg.MAX_LEN)

train_params = {'batch_size': cfg.TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 8
                }

test_params = {'batch_size': cfg.VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 8
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)

df = pd.read_csv(os.path.join(path, 'subgroup', 'val_preprocess.csv'))
for epoch in range(cfg.EPOCHS):
    if epoch == 1:
        for name, param in model.l1.encoder.named_parameters():
            param.requires_grad = True
    train(model, epoch, optimizer, training_loader)
    outputs = validation(model, testing_loader, df)

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    torch.save(model.state_dict(), os.path.join('checkpoint', f"{epoch}_model.ckpt"))
