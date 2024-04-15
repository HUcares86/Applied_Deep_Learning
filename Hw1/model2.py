# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate DLEnv
from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding
from torchcrf import CRF

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
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # 都放dropOut
        self.model = nn.LSTM(input_size=embeddings.shape[1], hidden_size=self.hidden_size
                             , num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional
                             , batch_first=True)
        self.model2 = nn.GRU(input_size=embeddings.shape[1], hidden_size=self.hidden_size
                        , num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional
                        , batch_first=True)
        # 把hidden_size變成所球機率 
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            # self.num_class會接收y有的種類數
            nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.num_class)
        )
        # =============================================
        # self.crf = CRF(self.num_class, batch_first=True)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return self.hidden_size * 2
        return self.hidden_size


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        # 接受collate_fn傳出的batch(文字已切割好轉成數字，並且已有padding好0)
        feature = batch['text']
        if 'intent' in batch.keys():
            label = batch['intent']
        # 把每個詞轉變成300維的vector
        embFeature = self.embed(feature)
        h_t, (h_n, c_n) = self.model(embFeature)

        # 只要抓最後一個layer的產出 若有bi變2倍 -> 抓取最後一個layer的兩個

        target_h = torch.cat((h_n[-1], h_n[-2]), axis=1) if self.bidirectional else h_n[-1]
        finalProp = self.classifier(target_h)
        # output 放入最終每段詞train出的150維機率表、選擇output字，與y算loss
        output = {}
        output['pred_prop'] = finalProp
        # 選擇機率最大的，讀取index即代表對應的tensor(因為用這樣的預測方式建立預測出的y算loss，故反覆訓練結果會讓y
        # 可以有這樣的對應
        output['pred'] = finalProp.max(1)[1].reshape(-1)
        pred = finalProp.max(1)[1].reshape(-1)
        if 'intent' in batch.keys():
            loss_fn = nn.CrossEntropyLoss()
            output['loss'] = loss_fn(output['pred_prop'], label)

        return output

class SeqTagger(SeqClassifier):

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        features, labels = batch['tokens'], batch['tags']
        embFeature = self.embed(features)

        packed_embFeature = nn.utils.rnn.pack_padded_sequence(embFeature, batch['len'], batch_first=True, enforce_sorted=False)
        self.model2.flatten_parameters()
        embFeature, _ = self.model2(packed_embFeature)
        embFeature, _ = nn.utils.rnn.pad_packed_sequence(embFeature, batch_first=True)  # [batch_size, max_len, hid_dim])

        # ================================================================
        # # CNN
        # embFeature = embFeature.permute(0, 2, 1) # [batch_size, embed_dim, max_len]
        # for conv in self.cnn:
        #     embFeature = conv(embFeature)
        # embFeature = embFeature.permute(0, 2, 1) # [batch_size, max_len, embed_dim]

        # ================================================================

        # h_t, (h_n, c_n) = self.model(embFeature)
        # 取最後一層的每個h_t(正反向都要)，取出後將其帶入classifier
        # 不可用128長度的seq 應要一個一個做
        # 把(128長度的)seq各自的1024個hidden改成3個class的機率
        h_t9Class = self.classifier(embFeature)
        output = {}
        # output['pred_prop'] -> 2 * 128 * 9 2個batch、128元素長度(128個token)、9個可能output的機率
        output['pred_prop'] = h_t9Class
        # 先轉出數字，記得前面data要補轉torch!!!
        # max後面輸入要max的維度->要選9個class類->dim=2
        # 選[1]代表選最大的機率對應的index(即為對應的class編號)
        output['pred'] = h_t9Class.max(2)[1]
        # ====================================================crf
        # output['pred'] = self.crf.decode(h_t9Class)
        # output['pred'] = torch.tensor(output['pred']).to("cuda")
        # ======================================================crf

        # 找目標index以便算loss
        batchNo = [torch.full((len,), i)for i, len in enumerate(batch['len'].long())]
        batchNoCom = torch.cat(batchNo)
        targetInd = [torch.arange(0, len)for len in (batch['len'].long())]
        targetIndCom = torch.cat(targetInd)
        output['y_true'] = []
        output['y_pred'] = []
        for i in range(len(batch['tokens'])):
            indOne = (batchNo[i], targetInd[i])
            output['y_true'].append(list(batch['tags'][indOne]))
            output['y_pred'].append(list(output['pred'][indOne]))
        # output['y_true'] = output['pred'][indOne] for indOne in (batchNo[i], targetInd[i])
        # output['y_true'] = list(labels[ind])
        # output['y_pred'] = list(output['pred'][ind])
        ind = (batchNoCom, targetIndCom)

        loss_fn = nn.CrossEntropyLoss()
        output['loss'] = loss_fn(output['pred_prop'][ind], labels[ind])

        #===================================================================crf
        # labels = labels[:, :output['pred_prop'].shape[1]]
        # batch['NotPad'] = batch['NotPad'][:, :output['pred_prop'].shape[1]]
        # batch['NotPad'] = batch['NotPad'].bool().to("cuda")
        # output['loss'] = -self.crf(output['pred_prop'], labels, batch['NotPad'])
        return output
