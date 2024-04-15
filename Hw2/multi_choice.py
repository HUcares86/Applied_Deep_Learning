from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import Multi_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from tqdm.auto import tqdm
from tqdm import trange
from multi_test import MultiTest


#tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")   
# 更好的 model


"""
4個paragraphs(多選)跟一個question(題目)街上, (放好101, 102後丟入padding function中)
看 哪個paragraphs屬於我們的question

"""

def tokenizeQuestion(data): 
    # 把Question tokenize
    question = []
    for data_ele in data:
        question.append(data_ele["question"])
    question_tokenized = tokenizer(question, add_special_tokens=False)
    return question_tokenized


def oneEpochTrain(model, scheduler, optimizer, train_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0    
    eval_loss_sum = 0
    eval_loss_count = 0
    eval_tok_cor = 0
    eval_tok_n = 0
    
    accumulation_steps = 5
    eval_steps = 100

    bar = tqdm(train_loader, desc="Train")  # 把train拿出來
    for batch in bar:
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['token_type_ids'] = batch['token_type_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['labels'] = batch['labels'].to(device)
        output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                       attention_mask=batch['attention_mask'], labels=batch['labels'])
        ans_label = torch.argmax(output.logits, dim=1)
        batch_cor = (batch['labels'].eq(ans_label.view_as(batch['labels']))).sum().long().item()
        tok_cor += batch_cor
        tok_n += batch['labels'].size(0)
        # ----------------------------------------------- acc Counting
        loss = output.loss
        # 是算整個batch的loss
        thisLossCount = batch['labels'].size(0)
        loss_sum += loss
        loss_count += thisLossCount
        loss.backward()
        step += 1
        # 改gradient Acc
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # -----------------------------------------------  eval
        eval_loss_sum += loss
        eval_loss_count += thisLossCount
        eval_tok_cor += batch_cor
        eval_tok_n += batch['labels'].size(0)
        if step % eval_steps == 0:
            accRate = tok_cor / (tok_n + 1E-8)
            avg_loss = loss_sum / loss_count
            print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            eval_accRate = eval_tok_cor / (eval_tok_n + 1E-8)
            eval_avg_loss = eval_loss_sum / eval_loss_count

            print('This {} step Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(eval_steps, eval_avg_loss, eval_accRate, eval_tok_cor, eval_tok_n))

            eval_loss_sum = 0
            eval_loss_count = 0
            eval_tok_cor = 0
            eval_tok_n = 0
        bar.set_postfix(lr=optimizer.param_groups[0]['lr'])
        
        # ----------------------------------------------- eval
        
    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

def oneEpochValid(model, valid_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    
    eval_steps = 100
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):
            # use gpu
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)

            output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                        attention_mask=batch['attention_mask'], labels=batch['labels'])
            ans_label = torch.argmax(output.logits, dim=1)
            # ----------------------------------------------- acc Counting
            batch_cor = (batch['labels'].eq(ans_label.view_as(batch['labels']))).sum().long().item()
            tok_cor += batch_cor
            tok_n += batch['labels'].size(0)
            # ----------------------------------------------- acc Counting
            # ----------------------------------------------- lossAndStep
            loss = output.loss
            # 是算整個batch的loss
            thisLossCount = batch['labels'].size(0)
            loss_sum += loss
            loss_count += thisLossCount
            step += 1
            if step % eval_steps == 0:
                accRate = tok_cor / (tok_n + 1E-8)
                avg_loss = loss_sum / loss_count
                print('Valid Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            # if step == 50:
            #     return

    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Valid Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

    return accRate


def saveCKPT(model):
    model_save_dir = "Multichoice_final_2"
    model.save_pretrained(model_save_dir)
    print("Model Save")


def main(args):
    # models
    model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-roberta-wwm-ext").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    #model = AutoModelForMultipleChoice.from_pretrained("bert-base-chinese").to(args.device)
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  
    
    #input data
    train_data_paths = args.data_dir / "train.json"
    valid_data_paths = args.data_dir / "valid.json"
    context_data_paths = args.data_dir / "context.json"

    train_data = json.loads(train_data_paths.read_text(encoding='utf-8'))
    valid_data = json.loads(valid_data_paths.read_text(encoding='utf-8'))
    context_data = json.loads(context_data_paths.read_text(encoding='utf-8'))

    # tokenize data
    train_data_tokenized = tokenizeQuestion(train_data)
    valid_data_tokenized = tokenizeQuestion(valid_data)
    context_tokenized = tokenizer(context_data, add_special_tokens=False)

    train_set = Multi_Dataset(train_data, train_data_tokenized, context_tokenized)
    valid_set = Multi_Dataset(valid_data, valid_data_tokenized, context_tokenized)

    # 查pin_memory用法
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # testData
    test_data_paths = args.data_dir / "test.json"
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))

    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 把scheduler改成對特定目標的sche
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    for epoch in range(args.num_epoch):
        oneEpochTrain(model, scheduler, optimizer, train_loader)
        oneEpochValid(model, valid_loader)
        saveCKPT(model)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)