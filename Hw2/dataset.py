import random

import transformers
from datasets import load_dataset, load_metric
import datasets
from typing import Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import torch
# from dataset import MultChoiceDataset
from transformers import BertTokenizerFast
import numpy as np
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
# from datasets import map
from torch.utils.data import Dataset


class Multi_Dataset(Dataset):

    def __init__(self, all_data, tokenized_questions, tokenized_paragraphs):

        # 把資料匯入
        self.all_data = all_data
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_seq_len = 420

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):

        sample_Dict = {}
        all_question = self.all_data[idx]
        tokenized_question = self.tokenized_questions[idx]

        # 最大長度512 question要讀滿
        self.max_paragraph_len = self.max_seq_len - len(tokenized_question.ids) - 3  # 一個開頭 一個分割 一個end

        tokenized_paragraph_list = [self.tokenized_paragraphs[idQ] for idQ in all_question["paragraphs"]]
        answer_label = 0
        # relavent 是我們training 的答案
        if "relevant" in all_question.keys():
            for j, idx in enumerate(all_question["paragraphs"]):
                if idx == all_question["relevant"]:
                    answer_label = j
        else:
            answer_label = 0
            
        # label = 0 or 1 or 2 or 3 ,代表在四篇文章中的list 裡面的深麽位置
        sample_Dict['labels'] = torch.tensor(answer_label)


        # 可考慮調整Q跟A2的max len
        input_ids_question = [101] + tokenized_question.ids + [102]
        input_ids_pgh_list = []
        for tokenized_paragraph in tokenized_paragraph_list:
            #我們看的文章長度
            #加結尾符號
            input_ids_pgh_list.append(tokenized_paragraph.ids[:self.max_paragraph_len] + [102])

        input_ids_list, token_type_ids_list, attention_mask_list = self.padding(input_ids_question, input_ids_pgh_list)

        # 輸入時需要的 accepted_keys 
        sample_Dict['input_ids'] = torch.tensor(input_ids_list)  # 1 * num_choice(4) * seq_len(after_padding)
        sample_Dict['token_type_ids'] = torch.tensor(token_type_ids_list)
        sample_Dict['attention_mask'] = torch.tensor(attention_mask_list)    # 做masking

        return sample_Dict

    # 欠 用token套件的padding package 做padding 把每個都補到一樣長
    def padding(self, input_ids_question, input_ids_pgh_list):
        padding_len_list = [self.max_seq_len - len(input_ids_question)
                            - len(input_ids_pgh) for input_ids_pgh in input_ids_pgh_list]
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        # print(padding_len_list)
        # print(input_ids_pgh_list)
        for input_ids_pgh, padding_len in zip(input_ids_pgh_list, padding_len_list):
            input_ids = input_ids_question + input_ids_pgh + [0] * padding_len
            input_ids_list.append(input_ids)

            token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_pgh) + [0] * padding_len
            token_type_ids_list.append(token_type_ids)

            attention_mask = [1] * (len(input_ids_question) + len(input_ids_pgh)) + [0] * padding_len
            attention_mask_list.append(attention_mask)

        return input_ids_list, token_type_ids_list, attention_mask_list


# def collate_fn(self, samples):
# 	batch = {}


class QA_Dataset(Dataset):

    def __init__(self, split, all_data, releIdxList, tokenized_questions, tokenized_paragraphs, tokenizer):

        self.split = split
        self.all_data = all_data
        if self.split == "test":
            self.releIdxList = releIdxList
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs

        self.tokenizer = tokenizer
        self.max_seq_len = 420
        self.doc_stride = 40

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):

        sample_Dict = {}
        all_question = self.all_data[idx]
        tokenized_question = self.tokenized_questions[idx]

        
        # 最大長度512 question要讀滿
        self.max_paragraph_len = self.max_seq_len - len(tokenized_question.ids) - 3  # 一個開頭 一個分割 一個end
        if self.split == "test":
            tokenized_paragraph = self.tokenized_paragraphs[all_question["paragraphs"][self.releIdxList[idx]]]

        else:
            releIdx = self.relevantIdxFind(all_question)
            tokenized_paragraph = self.tokenized_paragraphs[all_question["paragraphs"][releIdx]]

        if "answer" in all_question.keys():
            #因為我們沒有結束在哪裡但有給開頭跟答案長度所以要自己加起來
            ans_start = tokenized_paragraph.char_to_token(all_question["answer"]["start"])
            ans_end = tokenized_paragraph.char_to_token(len(all_question["answer"]["text"]) + all_question["answer"]["start"] - 1)
            sample_Dict['answer'] = all_question["answer"]["text"]

        else:
            # 給無意義的資訊
            ans_start = 0
            ans_end = 1
            sample_Dict['answer'] = ""

        # 做要開的paragraph在文章中的位置

		# 隨機start點即可，end配合start自然出現
		# pgh_start最左: max(ans_end - max_len, 0)
		# if 是ans_end - max_len -> end為ans_end
		# else 是min(0 + max_len, len(pgh) - 1)

        if self.split == "train":

            # train 時故意從答案的中間切割開始訓練，在隨機開一個window包住答案
            mid = (ans_start + ans_end) // 2
            answer_length = ans_end - ans_start + 1
            if (answer_length // 2) < (self.max_paragraph_len - (answer_length // 2)):
                rnd = random.randint(answer_length // 2, self.max_paragraph_len - answer_length // 2)
            else:
                rnd = self.max_paragraph_len // 2
            paragraph_start = max(0, min(mid - rnd, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # 可考慮調整Q跟A2的max len
            input_ids_question = [101] + tokenized_question.ids + [102]
            input_ids_pgh = tokenized_paragraph.ids[paragraph_start:paragraph_end] + [102]


            ans_start += len(input_ids_question) - paragraph_start
            ans_end += len(input_ids_question) - paragraph_start

            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_pgh)

            sample_Dict['input_ids'] = torch.tensor(input_ids)  # 1 * num_choice(4) * seq_len(after_padding)
            sample_Dict['token_type_ids'] = torch.tensor(token_type_ids)
            sample_Dict['attention_mask'] = torch.tensor(attention_mask)
            sample_Dict['start'] = torch.tensor(ans_start)
            sample_Dict['end'] = torch.tensor(ans_end)
        else:
            # pgh的資訊全部都必須考慮到，用窗格重疊的方式
            # window 重疊到就不用擔心答案在兩個 WINDOW 中間沒被看到回被切斷
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            # ex:8個窗口 -> 8種組合中找best
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]

                # ans_start += len(tokenized_question.ids) - i
                # ans_end += len(tokenized_question.ids) - i

                # Pad sequence and obtain inputs to model

                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            
            # squeeze; remove dimentions
            # print(sampleDict['input_ids'].shape)
            sample_Dict['input_ids'] = torch.tensor(input_ids_list).squeeze(
                dim=0)  # (1 -> 出去後有windows數*batch_size個) * num_choice(4) * seq_len(after_padding)
            # print('input_ids')
            # print(sampleDict['input_ids'].shape)
            # print(sampleDict['input_ids'].squeeze(dim=0).shape)
            sample_Dict['token_type_ids'] = torch.tensor(token_type_ids_list).squeeze(dim=0)
            sample_Dict['attention_mask'] = torch.tensor(attention_mask_list).squeeze(dim=0)

        return sample_Dict

    #用token套件的padding package 做padding
    def padding(self, input_ids_question, input_ids_pgh):
        # 還需要padding多少
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_pgh)
        input_ids = input_ids_question + input_ids_pgh + [0] * padding_len

        # token: paragraphs 要用 [1] ，question 要用 [0] 
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_pgh) + [0] * padding_len
        # token_type_ids_list.append(token_type_ids)

        # attention: paragraphs question 要用 [1] ， 因為那是我們關注的重點
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_pgh)) + [0] * padding_len
        # attention_mask_list.append(attention_mask)

        return input_ids, token_type_ids, attention_mask

    def relevantIdxFind(self, one_all_data):
        # 找答案
        # if "relevant" in one_all_data.keys():
        for j, idx in enumerate(one_all_data["paragraphs"]):
            if idx == one_all_data["relevant"]:
                return j
