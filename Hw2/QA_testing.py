from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import QA_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange
import numpy


class QATest:

    def __init__(self, test_data, context_data, relList, model, tokenizer, device):
        self.test_data = test_data
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        test_question_tokenized = self.tokenizeQuestion(test_data)
        context_tokenized = self.tokenizer(context_data, add_special_tokens=False)
        test_set = QA_Dataset("test", test_data, relList, test_question_tokenized, context_tokenized, tokenizer)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)


    def tokenizeQuestion(self, data):
        question = []
        for data_ele in data:
            question.append(data_ele["question"])
        question_tokenized = self.tokenizer(question, add_special_tokens=False)
        return question_tokenized

    def evaluate(self, batch, output):
        answer = ''
        max_prob = float('-inf')
        number_of_windows = batch['input_ids'].shape[0]

        for k in range(number_of_windows):
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)
            prob = start_prob + end_prob
            if prob > max_prob:
                if start_index > end_index:
                    continue
                else:
                    max_prob = prob

                    answer = self.tokenizer.decode(batch['input_ids'][k][start_index: end_index + 1])

        return answer.replace(' ', '')



    def testResult(self):
        # 做test_list


        answer_list = []
        for batch in tqdm(self.test_loader, desc="Test"):
            batch['input_ids'] = batch['input_ids'].squeeze(dim=0)
            batch['token_type_ids'] = batch['token_type_ids'].squeeze(dim=0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(dim=0)

            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
            with torch.no_grad():
                output_dict = self.model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'])
            answer = self.evaluate(batch, output_dict)
            answer_list.append(answer)

        return answer_list


