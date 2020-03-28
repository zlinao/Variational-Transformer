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
from collections import deque

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



def clean(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    return sentence
def read_langs(vocab):
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    train_context = np.load('data/empathetic-dialogue/sys_dialog_texts.train.npy')
    train_target = np.load('data/empathetic-dialogue/sys_target_texts.train.npy')
    train_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.train.npy')
    train_situation = np.load('data/empathetic-dialogue/sys_situation_texts.train.npy')

    dev_context = np.load('data/empathetic-dialogue/sys_dialog_texts.dev.npy')
    dev_target = np.load('data/empathetic-dialogue/sys_target_texts.dev.npy')
    dev_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.dev.npy')
    dev_situation = np.load('data/empathetic-dialogue/sys_situation_texts.dev.npy')
    
    test_context = np.load('data/empathetic-dialogue/sys_dialog_texts.test.npy')
    test_target = np.load('data/empathetic-dialogue/sys_target_texts.test.npy')
    test_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.test.npy')
    test_situation = np.load('data/empathetic-dialogue/sys_situation_texts.test.npy')

    data_train = {'context':[],'target':[],'emotion':[], 'situation':[]}
    data_dev = {'context':[],'target':[],'emotion':[], 'situation':[]}
    data_test = {'context':[],'target':[],'emotion':[], 'situation':[]}

    for context in train_context:
        u_list = deque([], maxlen=3)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(list(u_list))
    for target in train_target:
        target = clean(target)
        data_train['target'].append(target)
        vocab.index_words(target)
    for situation in train_situation:
        situation = clean(situation)
        data_train['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in train_emotion:
        data_train['emotion'].append(emotion)
    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['situation'])

    for context in dev_context:
        u_list = deque([], maxlen=3)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(list(u_list))
    for target in dev_target:
        target = clean(target)
        data_dev['target'].append(target)
        vocab.index_words(target)
    for situation in dev_situation:
        situation = clean(situation)
        data_dev['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)
    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['situation'])

    for context in test_context:
        u_list = deque([], maxlen=3)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_test['context'].append(list(u_list))
    for target in test_target:
        target = clean(target)
        data_test['target'].append(target)
        vocab.index_words(target)
    for situation in test_situation:
        situation = clean(situation)
        data_test['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in test_emotion:
        data_test['emotion'].append(emotion)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])
    return data_train, data_dev, data_test, vocab

def read_langs_persona(data_train, data_valid, data_test, vocab):
    with open('data/Persona/train_self_original.txt',encoding='utf-8') as f_train:
        context = deque([], maxlen=3)
        for line in f_train:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_train['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_train['target'].append(r)
                context.append(r)
                data_train['situation'].append(['dummy'])
                data_train['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    
    with open('data/Persona/valid_self_original.txt',encoding='utf-8') as f_valid:
        context = deque([], maxlen=3)
        for line in f_valid:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_valid['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_valid['target'].append(r)
                context.append(r)
                data_valid['situation'].append(['dummy'])
                data_valid['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    
    with open('data/Persona/test_self_original.txt',encoding='utf-8') as f_test:
        context = deque([], maxlen=3)
        for line in f_test:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_test['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_test['target'].append(r)
                context.append(r)
                data_test['situation'].append(['dummy'])
                data_test['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])
    return data_train, data_valid, data_test, vocab

def load_dataset():
    if(os.path.exists('data/persona_ed/dataset_preproc.p')):
        print("LOADING persona_ed")
        with open('data/persona_ed/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab  = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.CLS1_idx:"CLS1", config.Y_idx:"Y",
        9: 'key_surprised', 10: 'key_excited', 11: 'key_annoyed', 12: 'key_proud', 13: 'key_angry', 14: 'key_sad', 15: 'key_grateful', 16: 'key_lonely', 17: 'key_impressed', 18: 'key_afraid', 19: 'key_disgusted', 20: 'key_confident', 21: 'key_terrified', 22: 'key_hopeful',
         23: 'key_anxious', 24: 'key_disappointed', 25: 'key_joyful', 26: 'key_prepared', 27: 'key_guilty', 28: 'key_furious', 29: 'key_nostalgic', 30: 'key_jealous', 31: 'key_anticipating', 32: 'key_embarrassed', 33: 'key_content', 34: 'key_devastated', 35: 'key_sentimental', 36: 'key_caring', 37: 'key_trusting', 38: 'key_ashamed', 39: 'key_apprehensive', 40: 'key_faithful'})) 
        data_tra, data_val, data_tst, vocab = read_langs_persona(data_tra, data_val, data_tst, vocab)
        
        with open('data/persona_ed/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
            
    for i in range(3):
        print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

