from utils.data_loader import prepare_data_seq
from utils import config
from model.seq2seq import SeqToSeq
if config.v2:
    from model.SVT import CvaeTrans
else:
    from model.GVT import CvaeTrans
from model.common_layer import evaluate,evaluate_tra, count_parameters, make_infinite, get_kld
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import os
import time 
import numpy as np
import math
from tensorboardX import SummaryWriter

data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

if(config.test):
    print("Test model",config.model)
    if(config.model == "trs" or config.model == "cvaetrs"):
        model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    # elif(config.model == "cvaetrs"):
    #     model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "cvaenad"):
        model = CvaeNAD(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "seq2seq"):
        model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "cvae"):
        model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained, is_eval=True)
    model = model.eval()
    #loss_test, ppl_test, kld_test, bow_test, elbo_test, bleu_score_g, d1,d2,d3 = evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)
    get_kld(model, data_loader_tst ,ty="test", max_dec_step=50)
    exit(0)

if(config.model == "seq2seq"):
    model = SeqToSeq(vocab)
elif(config.model == "cvae"):
    model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained)
elif(config.model == "trs"):
    model = CvaeTrans(vocab,emo_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
elif(config.model == "cvaetrs"):
    model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)

elif(config.model == "cvaenad"):
    model = CvaeNAD(vocab,emo_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))

check_iter = 1000 if config.dataset=="empathetic" else 1000
if config.persona:
    check_iter = 1000

try:
    model = model.train()
    best_elbo = 1000
    patient = 0
    writer = SummaryWriter(log_dir=config.save_path)
    weights_best = deepcopy(model.state_dict())
    data_iter = make_infinite(data_loader_tra)
    for n_iter in tqdm(range(1000000)):
        if config.gradient_accumulation_steps>1:
            loss, ppl, kld, bow, elbo = model.train_n_batch([next(data_iter) for i in range(config.gradient_accumulation_steps)],n_iter)
        else:
            loss, ppl, kld, bow, elbo = model.train_one_batch(next(data_iter),n_iter)
        writer.add_scalars('loss', {'loss_train': loss}, n_iter)
        writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
        writer.add_scalars('kld', {'kld_train': kld}, n_iter)
        writer.add_scalars('bow', {'bow_train': bow}, n_iter)
        writer.add_scalars('elbo', {'elbo_train': elbo}, n_iter)
        if(config.noam):
            writer.add_scalars('lr', {'learning_rata': model.optimizer._rate}, n_iter)

        if((n_iter+1)%check_iter==0):
            model = model.eval()
            model.epoch = n_iter
            model.__id__logger = 0
            #evaluate_tra(model, data_loader_tra ,ty="valid", max_dec_step=50)
            loss_val, ppl_val, kld_val, bow_val, elbo_val, bleu_score_g, d1,d2,d3= evaluate(model, data_loader_val ,ty="valid", max_dec_step=50)
            writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
            writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
            writer.add_scalars('kld', {'kld_valid': kld_val}, n_iter)
            writer.add_scalars('bow', {'bow_valid': bow_val}, n_iter)
            writer.add_scalars('elbo', {'elbo_valid': elbo_val}, n_iter)
            model = model.train()
            best_elbo = elbo_val
            model.save_model(best_elbo,n_iter,ppl_val ,0,bleu_score_g,kld_val)
            weights_best = deepcopy(model.state_dict())

            if config.model=="trs":
                if config.dataset=="empathetic":
                    if n_iter>9000: break
                else:
                    if n_iter>17000: break
            else:
                if config.dataset=="empathetic":
                    if config.v2:
                        if n_iter>15000: break
                    else:
                        if n_iter>10000: break
                else:
                    if config.v2:
                        if n_iter>25000: break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

## TESTING
model.load_state_dict({ name: weights_best[name] for name in weights_best })
model.eval()
model.epoch = 100
loss_test, ppl_test, kld_test, bow_test, elbo_test, bleu_score_g, d1,d2,d3 = evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)

file_summary = config.save_path+"summary.txt"
with open(file_summary, 'w') as the_file:
    the_file.write("EVAL\tLoss\tPPL\tKLD\tELBO\tBleu_g\td1\td2\td3\n")
    the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("test",loss_test,ppl_test,kld_test, elbo_test,bleu_score_g,d1,d2,d3))
    