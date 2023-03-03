import os
import argparse
from tqdm import tqdm
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename='yrqUni.log', filemode='w')

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--dataPath", default='./data.csv', type=str)
parser.add_argument("--pathModel", default='./protbert', type=str)
parser.add_argument("--maxLenTrain", default=256, type=int)
parser.add_argument("--maxLenTest", default=512, type=int)
parser.add_argument("--UseAMP", default='UseAMP', type=str)
parser.add_argument("--sample", default=0, type=int)
parser.add_argument("--kFold", default=5, type=int)
parser.add_argument("--LabelSM", default=0.15, type=int)
parser.add_argument("--visWarn", default=0, type=int)
parser.add_argument("--WarmUp", default=1, type=int)
parser.add_argument("--PTWPlr", default=1e-10, type=float)
parser.add_argument("--NewWPlr", default=1e-3, type=float)
parser.add_argument("--lr", default=5e-6, type=float)
args = parser.parse_args()

if args.visWarn==0:
    transformers.utils.logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

if args.sample==0:
    data = pd.read_csv(args.dataPath)
if args.sample==1:
    data = pd.read_csv(args.dataPath).sample(1000)
if args.sample==2:
    data = pd.read_csv(args.dataPath)
    logging.info('raw data info {} {}'.format(data['label'].mean(),data.shape))
    gbr = data.groupby('label')
    gbr.groups
    typicalFracDict = {0:0.5, 1:0.5,}
    def typicalSampling(group, typicalFracDict):
        name = group.name
        frac = typicalFracDict[name]
        return group.sample(frac=frac)
    data = data.groupby('label', group_keys=False).apply(typicalSampling, typicalFracDict)
logging.info('sample data info {} {}\n'.format(data['label'].mean(),data.shape))
sample = list(data['feature'])
label = list(data['label'])

class DataSet(Dataset):
    def __init__(self, args=None, mode=None, sample=None, label=None):
        self.args = args
        self.mode = mode
        self.sample = sample
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pathModel)
    def __getitem__(self, idx):
        sample = self.sample[idx]
        label = self.label[idx]
        if self.mode=='train':
            inputs = self.tokenizer(sample, return_tensors='pt', padding='max_length', max_length=self.args.maxLenTrain, truncation=True)
        else:
            inputs = self.tokenizer(sample, return_tensors='pt', padding='max_length', max_length=self.args.maxLenTest, truncation=True)
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return {'input_ids':input_ids,
                'attention_mask':attention_mask, 
                'label':label}
    def __len__(self):
        return len(self.sample)

class UniModel(nn.Module):
    def __init__(self, args=None, hidden_dim=128, output_dim=2):
        super(UniModel, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(self.args.pathModel)
        self.lstm = nn.LSTM(input_size=self.bert_model.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True)
        self.drop_out = nn.Dropout(0.1)
        self.linear_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Softmax(dim=1)
    def forward(self, data):
        pooled_output = self.bert_model(data['input_ids'].to(torch.device(self.args.device)), data['attention_mask'].to(torch.device(self.args.device))).pooler_output 
        output, _ = self.lstm(pooled_output.unsqueeze(0))
        output = self.drop_out(output)
        output = self.linear_layer(output.squeeze(0))
        output = self.act(output)
        return output
    
def get_group_parameters(model=None, args=None, vis=None):
    params = list(model.named_parameters())
    no_decay = ['bias,','LayerNorm']
    other = ['lstm','linear_layer']
    no_main = no_decay + other
    param_group = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-2,'lr':args.PTWPlr},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],'weight_decay':0,'lr':args.PTWPlr},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],'weight_decay':0,'lr':args.NewWPlr},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],'weight_decay':1e-2,'lr':args.NewWPlr},
    ]
    if vis:
        logging.info('\nLayered learning rate para check pass {}'.format(set([n for n,p in params if not any(nd in n for nd in no_main)]+
                       [n for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)]+
                       [n for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ]+
                       [n for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)])==set([n for n,p in list(model.named_parameters())])))
    return param_group

kf = KFold(n_splits=args.kFold, random_state=0, shuffle=True)
k = 0
val_all = {'loss':[], 'auc':[]}
for train_index, test_index in kf.split(sample):
    val_infold = {'loss':[], 'auc':[]}
    trainData = DataSet(args, mode='train', sample=[sample[i] for i in train_index.tolist()], label=[label[i] for i in train_index.tolist()])
    testData = DataSet(args, mode='test', sample=[sample[i] for i in test_index.tolist()], label=[label[i] for i in test_index.tolist()])
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if args.UseAMP=='UseAMP':
        scaler = GradScaler()
    model = UniModel(args, hidden_dim=128, output_dim=2).to(torch.device(args.device))
    if args.LabelSM!=-1:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.LabelSM).to(torch.device(args.device))
    else:
        loss_fn = nn.CrossEntropyLoss().to(torch.device(args.device))
    optim.AdamW(get_group_parameters(model=model, args=args, vis=True),lr=1e-100,eps=1e-7)
    allBatchNum = 0
    for epoch in range(args.epochs):
        logging.info('------- {}F {}E -------'.format(k+1, epoch+1))
        model.train()
        total_train_loss = 0
        total_auc = 0
        for data in tqdm(trainLoader, desc='Train {}F {}E'.format(k+1, epoch+1)):
            allBatchNum = allBatchNum+1
            if allBatchNum<=(args.WarmUp*len(trainLoader)):
                optimizer = optim.AdamW(get_group_parameters(model=model, args=args, vis=False),lr=1e-100,eps=1e-7)
                WarmUpFlag1 = True
            else:
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
                WarmUpFlag2 = False
                if WarmUpFlag1!=WarmUpFlag2:
                    logging.info('WarmUp {}, WarmUp done at {} batch'.format(allBatchNum<=(args.WarmUp*len(trainLoader)), allBatchNum))
                WarmUpFlag1 = WarmUpFlag2
            if args.UseAMP=='UseAMP':
                with autocast():
                    out = model(data)
                    loss = loss_fn(out, data['label'].to(torch.device(args.device)))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()      
            else:
                out = model(data)
                loss = loss_fn(out, data['label'].to(torch.device(args.device)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_train_loss = total_train_loss+loss.item()
            fpr, tpr, thresholds = metrics.roc_curve(data['label'].detach().cpu().numpy(), out.detach().cpu().numpy()[:,1], pos_label=1)
            aucTrain = metrics.auc(fpr, tpr)
            total_auc = total_auc + aucTrain
        logging.info('train loss: {:.3f}'.format(total_train_loss/(len(trainLoader)+1)))
        logging.info('train auc: {:.3f}'.format(total_auc/(len(trainLoader)+1)))
        model.eval()
        total_test_loss = 0
        total_auc = 0
        with torch.no_grad():
            for data in tqdm(testLoader, desc='Test {}F {}E'.format(k+1, epoch+1)):
                if args.UseAMP=='UseAMP':
                    with autocast():
                        out = model(data)
                        loss = loss_fn(out, data['label'].to(torch.device(args.device)))
                total_test_loss = total_test_loss + loss.item()
                fpr, tpr, thresholds = metrics.roc_curve(data['label'].detach().cpu().numpy(), out.detach().cpu().numpy()[:,1], pos_label=1)
                aucTest = metrics.auc(fpr, tpr)
                total_auc = total_auc + aucTest
        val_infold['loss'].append(total_test_loss/(len(testLoader)+1))
        val_infold['auc'].append(total_auc/(len(testLoader)+1))
        logging.info('val loss: {:.3f}'.format(total_test_loss/(len(testLoader)+1)))
        logging.info('val auc: {:.3f}\n'.format(total_auc/(len(testLoader)+1)))        
    val_all['loss'].append(min(val_infold['loss']))
    val_all['auc'].append(max(val_infold['auc']))
    k = k + 1
logging.info('\nAll Done! Avg Val Loss{}, Avg Val AUC{}'.format(sum(val_all['loss'])/len(val_all['loss']),sum(val_all['auc'])/len(val_all['auc'])))
# yrqUni