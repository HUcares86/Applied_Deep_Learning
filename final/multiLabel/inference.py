import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import random
import util
import os
from argparse import ArgumentParser
from pathlib import Path
from config import Config

cfg = Config()

def fix_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", help="path")
    parser.add_argument("--model_path", help="path")
    parser.add_argument("--task", help="seen/unseen")
    parser.add_argument("--output", help="seen/unseen")
    args = parser.parse_args()
    fix_random_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = args.path
    task = args.task
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    model = util.BERTClass().to(device)
    model.load_state_dict(torch.load( args.model_path))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    df = pd.read_csv(os.path.join(path, 'subgroup', f"test_{task}_preprocess.csv"))
    df = df.fillna("")
    testing_set = util.CustomDataset( df, tokenizer, cfg.MAX_LEN)

    test_params = {'batch_size': cfg.VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 8
                    }
    
    testing_loader = DataLoader(testing_set, **test_params)
    
    output = util.pred_output( model, df, testing_loader, device )
    output.to_csv( output_file, index=False)