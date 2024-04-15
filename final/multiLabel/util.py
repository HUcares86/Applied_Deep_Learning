import transformers
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def pred_output(model, df, data_loader, device):
    user_id = {y: x for x, y in df['user_id'].items()}
    out = df[['user_id']].copy()
    out['subgroup'] = ""
    with torch.no_grad():
        for _, data in enumerate(tqdm(data_loader)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            _, outputs = torch.topk(outputs, k=50, sorted=True)
            for i in range(outputs.shape[0]):
                out['subgroup'][user_id[data['user_id'][i]]] = " ".join([str(x.item()) for x in outputs[i]])
    return out


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', return_dict=False)
        for name, param in self.l1.embeddings.named_parameters():
            param.requires_grad = False
        for name, param in self.l1.encoder.named_parameters():
            param.requires_grad = False
        # self.l1.pooler.dense = torch.nn.Sequential(
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(768, 256)
        # )
        self.out = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 92)
        )

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.out(output_1)
        return output


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.comment_text = dataframe.comment_text
        self.targets = dataframe.list
        self.user_id = dataframe.user_id
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        # comment_text = " ".join(comment_text.split())
        inputs = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        target = [float(x) for x in self.targets[index].split(" ")]

        return {
            'user_id': self.user_id[index],
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.float)
        }
