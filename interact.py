from utils.data_loader import prepare_data_seq
from utils import config
if config.v2:
    from model.SVT import CvaeTrans
else:
    from model.GVT import CvaeTrans
from model.seq2seq import SeqToSeq
from model.common_layer import evaluate,evaluate_tra, count_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import torch.utils.data as data
from tqdm import tqdm
import os
import time 
import numpy as np
import math
from collections import deque
DIALOG_SIZE = 3

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
    def __len__(self):
        return 1
    def __getitem__(self, index):
        # here we ignore index since we only have one input
        item = {}
        item["context_text"] = [x for x in self.data if x!="None"]
        X_dial = [config.CLS1_idx]
        X_mask = [config.CLS1_idx]
        for i, sentence in enumerate(item["context_text"]):
            X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence.split()]
            spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
            X_mask += [spk for _ in range(len(sentence.split()))]
        assert len(X_dial) == len(X_mask)
        item["context"] = X_dial
        item["mask"] = X_mask
        item["len"] = len(X_dial)
        return item

def collate_fn(data):
    
    input_batch = torch.LongTensor([data[0]["context"]])
    input_mask = torch.LongTensor([data[0]["mask"]])
    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        input_mask = input_mask.cuda()
    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor([data[0]["len"]])
    d["input_mask"] = input_mask
    d["program_label"] = torch.LongTensor([9]) #fake label
    if config.USE_CUDA:
        d["program_label"] = d["program_label"].cuda()
    return d 

def make_batch(inp,vacab):
    d = Dataset(inp,vacab)
    loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return iter(loader).next()

data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

if(config.model == "cvae"):
    model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained, is_eval=True)
else:
    model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
print('Start to chat')
context = deque(DIALOG_SIZE * ['None'], maxlen=DIALOG_SIZE)
while(True):
    msg = input(">>> ")
    if(len(str(msg).rstrip().lstrip()) != 0):

        context.append(str(msg).rstrip().lstrip())
        batch = make_batch(context, vocab)
        sent_g = model.decoder_greedy(batch,max_dec_step=30)
        print(">>>",sent_g[0])
        context.append(sent_g[0])
