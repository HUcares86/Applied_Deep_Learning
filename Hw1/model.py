
from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding
from torch import functional
class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.model = nn.LSTM(input_size=embeddings.shape[1], hidden_size=self.hidden_size
                             , num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional
                             , batch_first=True)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.num_class)
        )



    @property
    def encoder_output_size(self) -> int:

        if self.bidirectional:
            return self.hidden_size * 2
        return self.hidden_size


    def forward(self, batch) -> Dict[str, torch.Tensor]:

        feature = batch['text']
        if 'intent' in batch.keys():
            label = batch['intent']
        embFeatures = self.embed(feature)
        h_t, (h_n, c_n) = self.model(embFeatures)


        target_h = torch.cat((h_n[-1], h_n[-2]), axis=1) if self.bidirectional else h_n[-1]
        finalProps = self.classifier(target_h)

        output = {}
        output['pred_prop'] = finalProps
        output['pred'] = finalProps.max(1)[1].reshape(-1)
        pred = finalProps.max(1)[1].reshape(-1)
        if 'intent' in batch.keys():
            loss_fn = nn.CrossEntropyLoss()
            output['loss'] = loss_fn(output['pred_prop'], label)

        return output

class SeqTagger(SeqClassifier):

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        features, labels = batch['tokens'], batch['tags']
        embFeature = self.embed(features)
        packed_embFeature = nn.utils.rnn.pack_padded_sequence(embFeature, batch['len'], batch_first=True, enforce_sorted=False)
        self.model.flatten_parameters()
        embFeature, _ = self.model(packed_embFeature)
        embFeature, _ = nn.utils.rnn.pad_packed_sequence(embFeature, batch_first=True)  # [batch_size, max_len, hid_dim])
        h_t9Class = self.classifier(embFeature)
        output = {}
        output['pred_prop'] = h_t9Class
        output['pred'] = h_t9Class.max(2)[1]
        batchNos = [torch.full((len,), i)for i, len in enumerate(batch['len'].long())]
        batchNoCom = torch.cat(batchNos)
        targetInd = [torch.arange(0, len)for len in (batch['len'].long())]
        targetIndCom = torch.cat(targetInd)
        ind = (batchNoCom, targetIndCom)
        loss_fn = nn.CrossEntropyLoss()
        output['loss'] = loss_fn(output['pred_prop'][ind], labels[ind])
        return output

