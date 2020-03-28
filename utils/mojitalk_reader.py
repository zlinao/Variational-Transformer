import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(vocab):
    data_train = {'context':[],'target':[],'emotion':[]}
    data_dev = {'context':[],'target':[],'emotion':[]}
    data_test = {'context':[],'target':[],'emotion':[]}
    with open("data/mojitalk_data/vocab.ori", encoding='utf-8') as f:
        for word in f:
            vocab.index_word(word.strip())
    with open("data/mojitalk_data/train.ori", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_train['emotion'].append(line[0])
            data_train['context'].append(line[1:])
    with open("data/mojitalk_data/train.rep", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_train['target'].append(line)
    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion'])

    with open("data/mojitalk_data/dev.ori", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_dev['emotion'].append(line[0])
            data_dev['context'].append(line[1:])
    with open("data/mojitalk_data/dev.rep", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_dev['target'].append(line)
    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion'])

    with open("data/mojitalk_data/test.ori", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_test['emotion'].append(line[0])
            data_test['context'].append(line[1:])
    with open("data/mojitalk_data/test.rep", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_test['target'].append(line)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion'])

    return data_train, data_dev, data_test, vocab


def load_dataset():
    if(os.path.exists('data/mojitalk/dataset_preproc.p')):
        print("LOADING mojitalk")
        with open('data/mojitalk/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")

        data_tra, data_val, data_tst, vocab  = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.CLS1_idx:"CLS1", config.Y_idx:"Y", 9: 'key_😁', 10: 'key_😂', 11: 'key_😄', 12: 'key_😅', 13: 'key_😉', 14: 'key_😊', 15: 'key_😋', 16: 'key_😎', 17: 'key_😍', 18: 'key_😘', 19: 'key_☺', 20: 'key_😐', 21: 'key_😑', 22: 'key_😏', 23: 'key_😣', 24: 'key_😪', 25: 'key_😫', 26: 'key_😴', 27: 'key_😌', 28: 'key_😜', 29: 'key_😒', 30: 'key_😓', 31: 'key_😔', 32: 'key_😕', 33: 'key_😖', 34: 'key_😞', 35: 'key_😤', 36: 'key_😢', 37: 'key_😭', 38: 'key_😩', 39: 'key_😬', 40: 'key_😳', 41: 'key_😡', 42: 'key_😠', 43: 'key_😷', 44: 'key_😈', 45: 'key_💀', 46: 'key_🙈', 47: 'key_🙊', 48: 'key_🙅', 49: 'key_💁', 50: 'key_💪', 51: 'key_✌', 52: 'key_✋', 53: 'key_👌', 54: 'key_👍', 55: 'key_👊', 56: 'key_👏', 57: 'key_🙌', 58: 'key_🙏', 59: 'key_👀', 60: 'key_❤', 61: 'key_💔', 62: 'key_💕', 63: 'key_💖', 64: 'key_💙', 65: 'key_💜', 66: 'key_💟', 67: 'key_✨', 68: 'key_♥', 69: 'key_🎶', 70: 'key_🎧', 71: 'key_🔫', 72: 'key_💯'})) 
        with open('data/mojitalk/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', " ".join(data_tra['context'][i]))
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

