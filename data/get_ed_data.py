#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.
For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
Examples
--------
.. code-block:: shell
  python display_data.py -t babi:task1k:1
"""

import io
import re
import os
import sys
import json

import numpy as np
import pandas as pd
import dill as pickle

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import nltk
import spacy

class Lang:
    def __init__(self):
        self.unk_idx = 0
        self.pad_idx = 1
        self.sou_idx = 2
        self.eou_idx = 3

        self.word2index = {'__unk__': self.unk_idx, '__pad__': self.pad_idx, '__sou__': self.sou_idx, '__eou__': self.eou_idx}
        self.word2count = {'__unk__': 0, '__pad__': 0, '__sou__': 0, '__eou__': 0}
        self.index2word = {self.unk_idx: "__unk__", self.pad_idx: "__pad__", self.sou_idx: "__sou__", self.eou_idx: "__eou__"} 
        self.n_words = 4 # Count default tokens

        self.nlp = spacy.load("en_core_web_sm")
        # add special case rule
        special_case = [{spacy.symbols.ORTH: u"__eou__"}]
        self.nlp.tokenizer.add_special_case(u"__eou__", special_case)

    def __len__(self):
        return len(self.word2index)

    def tokenize(self, s):
        # return nltk.word_tokenize(s)
        return self.nlp.tokenizer(s)

    def addSentence(self, sentence):
        # for word in sentence.split(' '):
        for word in self.tokenize(sentence):
            self.addWord(word.text)

    def addSentences(self, sentences):
        for sentence in sentences:
            for word in self.tokenize(sentence):
                self.addWord(word.text)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def transform(self, sentences):
        # given unokenized sentences (or iterator), transform to idx mapping
        return [[self.word2index[token.text] for token in self.tokenize(sentence)] for sentence in sentences]

    def transform_one(self, sentence):
        try:
        # given unokenized sentence, transform to idx mapping
            return [self.word2index[token.text] for token in self.tokenize(sentence)]
        except KeyError as e:
            print(e)
            print(sentence)
            return []

    def reverse(self, sentences):
        # given transformed sentences, reverse it
        return [[self.index2word[idx] for idx in sentence] for sentence in sentences]

    def reverse_one(self, sentence):
        # given transformed sentence, reverse it
        return [self.index2word[idx] for idx in sentence]


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-ne', '--num_examples', type=int, default=10)
    parser.add_argument('-mdl', '--max_display_len', type=int, default=1000)
    parser.add_argument('--display_ignore_fields', type=str, default='agent_reply')
    parser.set_defaults(datatype='train:stream')
    return parser

def clean_msg(msg, msg_type):
    if 'labels' not in msg_type:
        msg = msg[3+len(msg_type):].strip()
    else:
        msg = msg[2+len(msg_type):-1].strip()
    msg = re.sub(r'"', '', msg)
    return msg

def display_data(opt, lang=None):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    dialog = []
    full_dialogs = []
    dialogs = []
    situation = ''
    situations = []
    emotion = ''
    emotions = []
    sys_emotions = []
    sys_situations = []
    targets = []
    usr_dialogs = [] # Listener 
    usr_targets = [] # Speaker
    sys_dialogs = [] # Speaker
    sys_targets = [] # Listener
    for i in range(opt['num_examples']):
        world.parley()

        # NOTE: If you want to look at the data from here rather than calling
        # world.display() you could access world.acts[0] directly
        message = world.display().split('\n')
        if situation != message[0] and i > 0:
            targets.append(dialog[-1])
            full_dialogs.append(dialog)
            dialogs.append(flatten(dialog[:-1]))
            sys_dialogs.append([dialog[0]])
            sys_targets.append(dialog[1])
            sys_emotions.append(clean_msg(emotion, 'emotion'))
            sys_situations.append(clean_msg(situation, 'situation'))
            
            if len(dialog) > 2:
                usr_dialogs.append(flatten(dialog[:2]))
                usr_targets.append(dialog[2])

            for t in range(2, len(dialog)):
                # Speaker
                if t % 2 == 0:
                    sys_dialogs.append(dialog[:t+1])
                # Listener
                else:
                    sys_targets.append(dialog[t])
                    sys_emotions.append(clean_msg(emotion, 'emotion'))
                    sys_situations.append(clean_msg(situation, 'situation'))
                    # Listener is not last turn
                    if len(dialog) > 2 and t < len(dialog) - 1:
                        usr_dialogs.append(flatten(dialog[t-1:t+1]))
                        usr_targets.append(dialog[t+1])
            dialog = []
            situations.append(clean_msg(situation, 'situation'))
            # if lang:
            #     add_sentence(lang, situations[-1])
            emotions.append(clean_msg(emotion, 'emotion'))
            # if lang:
            #     add_sentence(lang, emotions[-1])
    
        situation = message[0]
        emotion = message[1]
        # as_emotions.append(clean_msg(emotion, 'emotion'))
        dialog.append(clean_msg(message[6], 'empathetic_dialogues'))
        if lang:
            add_sentence(lang, dialog[-1])
        if 'train' not in opt['datatype']:
            dialog.append(clean_msg(message[7], 'eval_labels'))
        else:
            dialog.append(clean_msg(message[7], 'labels'))

        if lang:
            add_sentence(lang, dialog[-1])
        if world.epoch_done():
            print('EPOCH DONE')
            break

    try:
        # print dataset size if available
        print('[ loaded {} episodes with a total of {} examples ]'.format(
            world.num_episodes(), world.num_examples()
        ))
    except Exception:
        pass
    
    return situations, emotions, sys_emotions, full_dialogs, dialogs, targets, usr_dialogs, usr_targets, sys_dialogs, sys_targets, sys_situations

def flatten(dialog):
    return ' '.join(dialog)

def add_sentence(lang, sent):
    lang.addSentence(sent)

def transform_data(lang, situations, emotions, sys_emotions, dialogs, targets, usr_dialogs, usr_targets, sys_dialogs, sys_targets, sure_situations):
    # transform data to Lang index
    emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31
    }
    # situations = [lang.transform_one(situation) for situation in situations]
    # situation_lens = [len(situation) for situation in situations]
    emotions = [emo_map[emotion] for emotion in emotions]
    sys_emotions = [emo_map[emotion] for emotion in sys_emotions]
    dialogs = [lang.transform_one(dialog) for dialog in dialogs]
    dialog_lens = [len(dialog) for dialog in dialogs]
    usr_dialogs = [lang.transform_one(dialog) for dialog in usr_dialogs]
    usr_dialog_lens = [len(dialog) for dialog in usr_dialogs]
    sys_dialogs = [lang.transform_one(dialog) for dialog in sys_dialogs]
    sys_dialog_lens = [len(dialog) for dialog in sys_dialogs]
    # dialogs = [[lang.transform_one(utterance) for utterance in dialog] for dialog in dialogs]
    targets = [lang.transform_one(target) for target in targets]
    target_lens = [len(target) for target in targets]
    usr_targets = [lang.transform_one(target) for target in usr_targets]
    usr_target_lens = [len(target) for target in usr_targets]
    sys_targets = [lang.transform_one(target) for target in sys_targets]
    sys_target_lens = [len(target) for target in sys_targets]
    # sentences = [lang.transform_one(sentence) for sentence in sure_situations]
    # sentence_lens = [len(sentence) for sentence in sentences]
    return emotions, sys_emotions, dialogs, targets, usr_dialogs, usr_targets, sys_dialogs, sys_targets, dialog_lens, target_lens, usr_dialog_lens, usr_target_lens, sys_dialog_lens, sys_target_lens

def map_to_sentiment(situations, emotions):
    positives = set([
        'excited', 'proud', 'grateful', 'impressed', 'confident', 'hopeful', 'joyful', 'prepared', 'anticipating', 'content', 'caring', 'trusting', 'faithful'
    ])
    negatives = set([
        'annoyed', 'angry', 'sad', 'lonely', 'afraid', 'disgusted', 'terrified', 'anxious', 'disappointed', 'guilty', 'furious', 'jealous', 'embarrassed', 'devastated', 'ashamed', 'apprehensive'
    ])

    sure_situations = []
    unsure_situations = []
    sentiments = []
    for i, (situation, emotion) in enumerate(zip(situations, emotions)):
        if emotion in positives:
            sure_situations.append(situation)
            sentiments.append(1)
        elif emotion in negatives:
            sure_situations.append(situation)
            sentiments.append(0)
        else:
            # drop examples that are neutral, and return them separately
            unsure_situations.append(situation)
    
    return sure_situations, unsure_situations, sentiments


def save_npy(data, path):
    np.save(path+'.npy', data)

def load_npy(path):
    return np.load(path+'.npy')

def save_pkl(data, path):
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_pkl(path):
    with open(path+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def load_vectors(fname, vocabulary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embedding_matrix = np.random.uniform(-0.01, 0.01, ((len(vocabulary)), d))
    print(embedding_matrix.shape)
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocabulary.keys():
            embedding_matrix[vocabulary[tokens[0]]] = np.array(list(map(float, tokens[1:])))
    print(embedding_matrix.shape)

    return embedding_matrix


if __name__ == '__main__':
    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists('data/prep/empathetic-dialogue/lang.pkl'):
        print("Creating vocab from full text")
        lang = Lang()
    else:
        print("Loading vocab")
        sys.argv = ['']
        lang = load_pkl('data/prep/empathetic-dialogue/lang')
        print(len(lang))

    situations, emotions, sys_emotions, full_dialogs, dialogs, targets, usr_dialogs, usr_targets, sys_dialogs, sys_targets, sys_situations = display_data(opt, lang)
    trc_dialogs, trc_targets = usr_dialogs, usr_targets
    print(len(lang))
    print(len(usr_dialogs), len(usr_targets))
    print(len(sys_dialogs), len(sys_targets), len(sys_emotions))
    print(situations[0])
    print(emotions[0])
    print(sys_emotions[0])
    print(sys_emotions[1])
    print(dialogs[0])
    print(targets[0])
    print(trc_dialogs[0])
    print(trc_targets[0])
    print(trc_dialogs[1])
    print(trc_targets[1])
    print(sys_dialogs[0])
    print(sys_targets[0])
    print(sys_dialogs[1])
    print(sys_targets[1])
    
    split = re.sub(r':ordered', '', opt['datatype'])
    split = re.sub(r':stream', '', split)
    split = re.sub(r'valid', 'dev', split)
    print(split)
    save_path = 'data/prep/empathetic-dialogue/{}.{}' # name
    save_npy(full_dialogs, save_path.format('full_dialog_texts', split))
    save_npy(usr_dialogs, save_path.format('usr_dialog_texts', split))
    save_npy(usr_targets, save_path.format('usr_target_texts', split))
    save_npy(sys_dialogs, save_path.format('sys_dialog_texts', split))
    save_npy(sys_targets, save_path.format('sys_target_texts', split))
    save_npy(sys_emotions, save_path.format('sys_emotion_texts', split))
    save_npy(sys_situations, save_path.format('sys_situation_texts', split))
    save_path = 'data/prep/empathetic-dialogue/{}' # name
    save_pkl(lang, save_path.format('lang'))
