import random
import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm.notebook import tqdm
import copy
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import argparse


pl.seed_everything(42)

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')


# load dataset
src_path = './funcom_processed/functions.json'
com_path = './funcom_processed/comments.json'
id_path = './funcom_processed/fid_pid.json'

with open(src_path, 'r') as fp:
  src = json.load(fp, parse_int=True)

with open(com_path, 'r') as fp:
  com = json.load(fp)

with open(id_path, 'r') as fp:
  id = json.load(fp)

id = [k for k in id.keys()]
source_comments_data = [(src[i],com[i]) for i in id]
source_comments_data = random.sample(source_comments_data, 10000)


train_set, val_set = torch.utils.data.random_split(source_comments_data, [int(len(source_comments_data)*0.8), int(len(source_comments_data)*0.2)])

train_dataset = SourceDataset(t5_tokenizer,train_set)
validation_dataset = SourceDataset(t5_tokenizer,val_set)


class SourceDataset(Dataset):
    def __init__(self, tokenizer, scr_com, max_len_inp=96,max_len_out=96):
        self.scr_com = scr_com
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount =0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def _clean_src_com(self, text):
        text = text.replace('\n','')
        text = text.replace('\t', '')
        text = text.replace('/**', '')
        text = text.replace('*/', '')
        text = text.replace('*', '')
        text = text.replace('<code>','')
        text = text.replace('</code>','')
        return text

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for inputs,outputs in tqdm(self.scr_com):
          input_sent = "comment: " + self._clean_src_com(inputs)
          ouput_sent = "commented: " + self._clean_src_com(outputs)

          # tokenize inputs
          tokenized_inputs = self.tokenizer.batch_encode_plus(
              [input_sent], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
          )
          # tokenize targets
          tokenized_targets = self.tokenizer.batch_encode_plus(
              [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
          )

          self.inputs.append(tokenized_inputs)
          self.targets.append(tokenized_targets)

class T5FineTuner(pl.LightningModule):
    def __init__(self,hparams, t5model, t5tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = t5model
        self.tokenizer = t5tokenizer


    def forward( self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
         
         return outputs


    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss",loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size,num_workers=4)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size,num_workers=4)



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


args_dict = dict(
    batch_size=1,
)

args = argparse.Namespace(**args_dict)


model = T5FineTuner(args,t5_model,t5_tokenizer)

trainer = pl.Trainer(max_epochs = 3, gpus=1,progress_bar_refresh_rate=30)

trainer.fit(model)


print("TRAINING FINISHED")