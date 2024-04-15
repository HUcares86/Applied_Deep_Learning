from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    def collate_fn(self, samples: List[Dict]) -> Dict:
        a = samples
        batch = {}
        batch['text'] = [OneSample['text'].split() for OneSample in samples]
        batch['text'] = [self.vocab.encode_batch(batch['text'], self.max_len)]
        batch['text'] = torch.tensor(batch['text']).reshape(-1, self.max_len)
        batch['id'] = [OneSample['id'] for OneSample in samples]
        if 'intent' in samples[0].keys():
            batch['intent'] = torch.tensor([self.label2idx(OneSample['intent']) for OneSample in samples])

        return batch


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]



class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100
    def collate_fn(self, samples):
        batch = {}

        batch['tokens'] = [oneSample['tokens'] for oneSample in samples]
        batch['len'] = torch.tensor([min(len(token), self.max_len) for token in batch['tokens']])
        batch['tokens'] = [self.vocab.encode_batch(batch['tokens'], self.max_len)]
        batch['tokens'] = torch.tensor(batch['tokens']).reshape(-1, self.max_len)

        if 'tags' in samples[0]:

            batch['tags'] = [oneSample['tags'] for oneSample in samples]

            batch['tags'] = [[self.label2idx(tag) for tag in tagL] for tagL in batch['tags']]
            batch['tags'] = pad_to_len(batch['tags'], self.max_len, 0)

            batch['tags'] = torch.tensor(batch['tags']).reshape(-1, self.max_len)

        else:
            batch['tags'] = torch.tensor([[0] * self.max_len] * len(samples))

        batch['id'] = [oneSample['id'] for oneSample in samples]

        batch['NotPad'] = batch['tokens'].gt(0).float()

        return batch

