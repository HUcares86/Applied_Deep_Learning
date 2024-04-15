from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange
from multi_test import MultiTest
from QA_testing import QATest
import csv
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
import json
from dataset import QA_Dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
#tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
# 更好的 model

# 其實要做的一些步驟跟multi_choice很像， 都要做tokenize等等

def tokenizeQuestion(data):
    questions = []
    for data_ele in data:
        questions.append(data_ele["question"])
    # questions = [[train_data_ele["question"] * 4 for train_data_ele in train_data]
    questions_tokenized = tokenizer(questions, add_special_tokens=False)
    return questions_tokenized



def oneEpochTrain(model, optimizer, scheduler, train_loader):
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
    eval_steps = 200


    # save lost and EM parameter
    totalLossList = []
    difLossList = []
    totalEMList = []
    difEMList = []


    bar = tqdm(train_loader, desc="Train")
    for batch in bar:
        # use gpu
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['token_type_ids'] = batch['token_type_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['start'] = batch['start'].to(device)
        batch['end'] = batch['end'].to(device)

        output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                       attention_mask=batch['attention_mask'], start_positions=batch['start'], end_positions=batch['end'])

        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        # ----------------------------------------------- acc Counting
        # ----------------------------------------------- lossAndStep

        batch_cor = ((start_index == batch['start']) & (end_index == batch['end'])).sum().long().item()
        tok_cor += batch_cor
        tok_n += batch['start'].size(0)

        loss = output.loss
        # 是算整個batch的loss
        this_Loss_Count = batch['start'].size(0)
        loss_sum += loss
        loss_count += this_Loss_Count
        loss.backward()
        step += 1
        # 改gradient Acc
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        eval_loss_sum += loss
        eval_loss_count += this_Loss_Count
        eval_tok_cor += batch_cor
        eval_tok_n += batch['input_ids'].size(0)
        
        if step % eval_steps == 0:
            accRate = tok_cor / (tok_n + 1E-8)
            avg_loss = loss_sum / loss_count
            totalEMList.append(accRate)
            totalLossList.append(avg_loss)
            print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            eval_accRate = eval_tok_cor / (eval_tok_n + 1E-8)
            eval_avg_loss = eval_loss_sum / eval_loss_count
            difEMList.append(eval_accRate)
            difLossList.append(eval_avg_loss)
            print('This {} step Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(eval_steps, eval_avg_loss, eval_accRate, eval_tok_cor, eval_tok_n))

            eval_loss_sum = 0
            eval_loss_count = 0
            eval_tok_cor = 0
            eval_tok_n = 0

        bar.set_postfix(lr=optimizer.param_groups[0]['lr'])

    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))
    print(totalLossList, difLossList)
    return totalLossList, difLossList, totalEMList, difEMList


def evaluate(batch, output):

    answer = ''
    max_prob = float('-inf')
    num_of_windows = batch['input_ids'].shape[0]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            if start_index > end_index:
                continue
            else:
                max_prob = prob
                answer = tokenizer.decode(batch['input_ids'][k][start_index: end_index + 1])
    return answer.replace(' ', '')


def oneEpochValid(model, valid_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    eval_steps = 50

    # batch size只有1
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):

            batch['input_ids'] = batch['input_ids'].squeeze(dim=0)
            batch['token_type_ids'] = batch['token_type_ids'].squeeze(dim=0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(dim=0)

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)

            # 可以不給ans去跑model!
            output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                        attention_mask=batch['attention_mask'])

            pred_answer = evaluate(batch, output)
            batch_cor = pred_answer == batch["answer"][0]

            tok_cor += batch_cor
            tok_n += 1
            # print(tok_n)
            
            step += 1
            if step % eval_steps == 0:
                accRate = tok_cor / (tok_n + 1E-8)
                # avg_loss = loss_sum / loss_count
                print('Valid Acc: {:6.4f} ({}/{})'.format(accRate, tok_cor, tok_n))

    accRate = tok_cor / (tok_n + 1E-8)
    # avg_loss = loss_sum / loss_count
    print('Valid Acc: {:6.4f} ({}/{})'.format(accRate, tok_cor, tok_n))

    return accRate

def saveCKPT(model):
    # model_save_dir = "saved_QA_model_Train"
    model_save_dir = "QA_final_2"
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

    train_data_tokenized = tokenizeQuestion(train_data)
    valid_data_tokenized = tokenizeQuestion(valid_data)
    context_tokenized = tokenizer(context_data, add_special_tokens=False)

    train_set = QA_Dataset("train", train_data, 0, train_data_tokenized, context_tokenized, tokenizer)
    valid_set = QA_Dataset("valid", valid_data, 0, valid_data_tokenized, context_tokenized, tokenizer)

    # 查pin_memory用法
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)

    # testData
    test_data_paths = args.data_dir / "test.json"
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))

    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.92)

    for epoch in range(args.num_epoch):
        totalLossList, difLossList, totalEMList, difEMList = oneEpochTrain(model, optimizer, scheduler, train_loader)
        # config = BertConfig.from_pretrained(model_path)
        oneEpochValid(model, valid_loader)
        saveCKPT(model)
        print(totalLossList)
        for i in range(len(totalLossList)):
            totalLossList[i] = totalLossList[i].item()
        for i in range(len(difLossList)):
            difLossList[i] = difLossList[i].item()
        print(totalLossList)
        for i in range(len(totalEMList)):
            totalEMList[i] = totalEMList[i]
        for i in range(len(difEMList)):
            difEMList[i] = difEMList[i]
        with open("total_Loss_1{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['total_Loss'])
            writer.writerow(totalLossList)

        with open("dif_Loss_1{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['dif_Loss'])
            writer.writerow(difLossList)
        
        with open("total_EM_1{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['total_EM'])
            writer.writerow(totalEMList)

        with open("dif_EM_1{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['dif_EM'])
            writer.writerow(difEMList)

    # 讀Multi model後出List結果，再把list結果套入
    multiModel = AutoModelForMultipleChoice.from_pretrained("./Multichoice_final_2").to(args.device)


    # # multiModel = multiModel.load
    print("loading_multi_test")
    multTestProcess = MultiTest(test_data, context_data, 1, multiModel, tokenizer, args.device)
    relList = multTestProcess.testResult()
    print("predicting_QA_test")
    QATestProcess = QATest(test_data, context_data, relList, model, tokenizer, args.device)
    answerList = QATestProcess.testResult()

    # 存答案!-----------------------------------
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
        default="./Data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # # loss
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

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