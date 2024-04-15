from numpy.random.mtrand import rand
import csv
import argparse
import numpy as np
import random
import glob
import os
import tqdm
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config
from models import hahow_net

cfg = Config()
if cfg.challenge == 'seen':
    import data_utils
else:
    import data_utils_unseen as data_utils


@torch.no_grad()
def main():

    cfg = Config()

    model = torch.load(cfg.model_path)
    model = model.to(cfg.device)
    model.eval()

    csv_file = open(cfg.output_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['user_id', 'course_id'])

    test_data = data_utils.HaHowTestData(path=cfg.test_file)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    id2cid = pd.read_csv(cfg.id2cid_file, usecols=['course_id'])
    id2cid = list(id2cid['course_id'])

    for inputs, avoid_courses, user_ids in tqdm.tqdm(test_loader):
        inputs = inputs.to(cfg.device)
        avoid_courses = avoid_courses.to(cfg.device)

        outputs = model(inputs)
        outputs = outputs + avoid_courses * -999

        top50_val, top50_idx = torch.topk(outputs, 50, dim=1)
        top50_idx = top50_idx.cpu().tolist()
        for i in range(inputs.shape[0]):
            top_courses = ' '.join([id2cid[x] for x, y in zip(top50_idx[i], top50_val[i]) if y > cfg.select_threshold])
            csv_writer.writerow([user_ids[i], top_courses])

    csv_file.close()


if __name__ == '__main__':
    main()
