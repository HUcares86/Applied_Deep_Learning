
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging
# from torch.cuda.amp import autocast
import csv

logger = logging.get_logger(__name__)


class MySeq2SeqTrainer(Trainer):
    # *args代表一堆參數(沒預設等於)，**kwargs代表有等於的參數
    # 設定成員變量
    def __init__(self, *args, eval_examples=None, gen_config=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.gen_config = gen_config
        self.post_process_function = post_process_function

    # 放入eval_dataset(可給None直接用self.eval_dataset)
    # 做data_loader(放入data_set, 給collate_fn)
    # 放入prediction_loop做預測，prediction_loop中吃data_loader
    # 做enumerate(dataloader)，用prediction_step函數出loss, generated_tokens(logit機率), labels
    # 預測完後把結果丟入post_process_function，出處理好的pred, label(用EvalPrediction Class包裝)
    # 調用compute_metrics讀取EvalPrediction出metrics
    # 最後回傳metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        print("dataloader ok!")
        # 欠修prediction_step
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        print("compute_metrics = None ok!")
        try:
            output = self.prediction_loop(eval_dataloader, "Evaluation", prediction_loss_only=None
                                        , metric_key_prefix="eval")
            print("output ok!")
            # prefix用在寫檔案!
            eval_pred = self.post_process_function(self.eval_examples, output, prefix='eval')
            print("post_process_function ok!")
        finally:
            self.compute_metrics = compute_metrics
            print("self.compute_metrics = compute_metrics ok!")
        


        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        preds = [pred for pred in preds]
        labels = [label for label in labels]
        # print("preds")
        # print(preds)
        # print("labels")
        # print(labels)
        with open("./preds4.csv", 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['pred'])
            writer.writerow(preds)
        print("preds.csv saved!")
        with open("./labels4.csv", 'w', encoding='utf-8-sig', newline='') as f2:
            writer = csv.writer(f2, delimiter=',')
            writer.writerow(['labels'])
            writer.writerow(labels)
        print("labels.csv saved!")
        
        print("start compute_metrics fun out")
        metric = self.compute_metrics(eval_pred)
        print("metric = self.compute_metrics(eval_pred) ok!")
        print(metric)
        return metric

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:

        test_dataloader = self.get_eval_dataloader(test_dataset)
        # 欠修prediction_step，若不把metrics設為None會自動調用
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        output = self.prediction_loop(test_dataloader, "Test", prediction_loss_only=None
                                      , metric_key_prefix="")
        # prefix用在寫檔案!
        eval_pred = self.post_process_function(self.eval_examples, output, prefix='test')
        self.compute_metrics = compute_metrics

        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        preds = [pred for pred in preds]
        labels = [label for label in labels]
        with open("./preds_do_sample.csv", 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['pred'])
            writer.writerow(preds)
            print("preds.csv saved!")
        with open("./labels_do_sample.csv", 'w', encoding='utf-8-sig', newline='') as f2:
            writer = csv.writer(f2, delimiter=',')
            writer.writerow(['labels'])
            writer.writerow(labels)
        print("labels.csv saved!")


        metric = self.compute_metrics(eval_pred)


        return PredictionOutput(predictions=eval_pred.predictions, label_ids=eval_pred.label_ids, metrics=metric)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:


        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # 產生generated_tokens，其中放入inputs["input_ids"],
        # attention_mask=inputs["attention_mask"], gen_config (查self.model.generate:決定生出機率後decode的方式???)
        # generate是將input代入train好的model並給不同的decode方式(用token的方式表現decode)

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **self.gen_config
        )
        # 可設條件選擇是否pad
        generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.gen_config["max_length"] + 1)

        with torch.no_grad():
            # if self.use_cuda_amp:
            #     with autocast():
            #         outputs = model(**inputs)
            # else:
            #     outputs = model(**inputs)
            # 做output取loss
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                # 查什麼是label_smoother
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    # 查detach
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < self.gen_config["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, self.gen_config["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
