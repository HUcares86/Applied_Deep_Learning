from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import QA_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange
from multi_test import MultiTest
from QA_test import QATest
import csv


from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering

def main(args):
    test_data_paths = args.test_file
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))
    context_data_paths = args.data_dir / "context.json"
    context_data = json.loads(context_data_paths.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    multiModel = AutoModelForMultipleChoice.from_pretrained(args.Multi_model).to(args.device)
    qaModel = AutoModelForQuestionAnswering.from_pretrained(args.QA_model).to(args.device)
    print("loading_multi_test")
    multTestProcess = MultiTest(test_data, context_data, args.batch_size, multiModel, tokenizer, args.device)
    relList = multTestProcess.testResult()
    print("predicting_QA_test")
    QATestProcess = QATest(test_data, context_data, relList, qaModel, tokenizer, args.device)
    answerList = QATestProcess.testResult()
    all_ids = []
    for allQues in test_data:
        all_ids.append(allQues['id'])
    with open(args.pred_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id','answer'])
        writer.writerows(zip(all_ids, answerList))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/test.json",
    )
    
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/",
    )

    parser.add_argument(
        "--Multi_model",
        type=Path,
        help="Directory to the dataset.",
        default="./Multichoice_final_2",
    )

    parser.add_argument(
        "--QA_model",
        type=Path,
        help="Directory to the dataset.",
        default="./QA_final_2",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=4e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=1)

    parser.add_argument("--pred_file", type=Path, default="final_2.csv")


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)