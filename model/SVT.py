### TAKEN FROM https://github.com/kolloldas/torchnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, VarDecoderLayer1, MultiHeadAttention, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, _get_attn_self_mask,  get_input_from_batch, get_output_from_batch, gaussian_kld,SoftmaxOutputLayer
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score

class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y
class VarDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, vocab_size, max_length=200, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(VarDecoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 vocab_size,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.vardec = nn.Sequential(*[VarDecoderLayer1(*params) for l in range(config.num_var_layers)])
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers- config.num_var_layers)])
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm_1 = LayerNorm(hidden_size)
        self.layer_norm_2 = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, posterior, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        dec_mask_p = mask_trg
        #Add input dropout

        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
            
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Run decoder
        y, _, _, attn_dist, _ , means, log_vars, logits_probs= self.vardec((x, encoder_output, posterior, [], (mask_src,dec_mask,dec_mask_p),{"prior":[],"posterior":[]},{"prior":[],"posterior":[]},[]))
        y = self.layer_norm_1(y)
        y, _, attn_dist, _ = self.dec((y, encoder_output, [], (mask_src,dec_mask)))
        y = self.layer_norm_2(y)
        if posterior:
            #print(means["prior"].shape)
            means["prior"] = torch.cat(means["prior"], 0)
            means["posterior"] = torch.cat(means["posterior"], 0)
            log_vars["prior"] = torch.cat(log_vars["prior"], 0)
            log_vars["posterior"] = torch.cat(log_vars["posterior"], 0)
            
        return y, attn_dist, means, log_vars, logits_probs

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist            
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit,dim=-1)



class CvaeTrans(nn.Module):

    def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(CvaeTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        
        # self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
        #                         total_key_depth=config.depth, total_value_depth=config.depth,
        #                         filter_size=config.filter,universal=config.universal)
        self.decoder = VarDecoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth,total_value_depth=config.depth,
                                filter_size=config.filter, vocab_size = self.vocab_size)

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        # if config.multitask:
        #     self.emo = SoftmaxOutputLayer(config.hidden_dim,emo_number)
        #     self.emo_criterion = nn.NLLLoss()
        
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            #self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            
        if (config.USE_CUDA):
            self.cuda()
        if is_eval:
            self.eval()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
            if(config.noam):
                self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
                if config.USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):

        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            #'r_encoder_state_dict': self.r_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        if(train):
            if(config.noam):
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)

        meta = self.embedding(batch["program_label"])
        if config.dataset=="empathetic":
            meta = meta-meta
        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist, mean, log_var, probs= self.decoder(self.embedding(dec_batch_shift)+meta.unsqueeze(1),encoder_outputs, True, (mask_src,mask_trg))
        ## compute output dist
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        sbow = dec_batch #[batch, seq_len]
        seq_len = sbow.size(1)
        loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        if config.model=="cvaetrs":
            loss_aux = 0
            for prob in probs:
                sbow_mask = _get_attn_subsequent_mask(seq_len).transpose(1,2)
                sbow.unsqueeze(2).repeat(1,1,seq_len).masked_fill_(sbow_mask,config.PAD_idx)#[batch, seq_len, seq_len]

                loss_aux+= self.criterion(prob.contiguous().view(-1, prob.size(-1)), sbow.contiguous().view(-1))
            kld_loss = gaussian_kld(mean["posterior"], log_var["posterior"],mean["prior"], log_var["prior"])
            kld_loss = torch.mean(kld_loss)
            kl_weight = min(math.tanh(6 * iter/config.full_kl_step - 3) + 1, 1)
            #kl_weight = min(iter/config.full_kl_step, 1) if config.full_kl_step >0 else 1.0
            loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux
            elbo = loss_rec+kld_loss
        else:
            loss = loss_rec
            elbo = loss_rec
            kld_loss = torch.Tensor([0])
            loss_aux = torch.Tensor([0])
        if(train):
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), loss_aux.item(), elbo.item()

    def train_n_batch(self, batchs, iter, train=True):
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        for batch in batchs:
            enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
            dec_batch, _, _, _, _ = get_output_from_batch(batch)
            ## Encode
            mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            emb_mask = self.embedding(batch["input_mask"])
            encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)

            meta = self.embedding(batch["program_label"])
            if config.dataset=="empathetic":
                meta = meta-meta
            # Decode
            sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
            if config.USE_CUDA: sos_token = sos_token.cuda()
            dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

            mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

            pre_logit, attn_dist, mean, log_var, probs= self.decoder(self.embedding(dec_batch_shift)+meta.unsqueeze(1),encoder_outputs, True, (mask_src,mask_trg))
            ## compute output dist
            logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
            ## loss: NNL if ptr else Cross entropy
            sbow = dec_batch #[batch, seq_len]
            seq_len = sbow.size(1)
            loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            if config.model=="cvaetrs":
                loss_aux = 0
                for prob in probs:
                    sbow_mask = _get_attn_subsequent_mask(seq_len).transpose(1,2)
                    sbow.unsqueeze(2).repeat(1,1,seq_len).masked_fill_(sbow_mask,config.PAD_idx)#[batch, seq_len, seq_len]

                    loss_aux+= self.criterion(prob.contiguous().view(-1, prob.size(-1)), sbow.contiguous().view(-1))
                kld_loss = gaussian_kld(mean["posterior"], log_var["posterior"],mean["prior"], log_var["prior"])
                kld_loss = torch.mean(kld_loss)
                kl_weight = min(math.tanh(6 * iter/config.full_kl_step - 3) + 1, 1)
                #kl_weight = min(iter/config.full_kl_step, 1) if config.full_kl_step >0 else 1.0
                loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux
                elbo = loss_rec+kld_loss
            else:
                loss = loss_rec
                elbo = loss_rec
                kld_loss = torch.Tensor([0])
                loss_aux = torch.Tensor([0])
            loss.backward()
            # clip gradient
        nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), loss_aux.item(), elbo.item()

    def decoder_greedy(self, batch, max_dec_step=50):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        meta = self.embedding(batch["program_label"])
        if config.dataset=="empathetic":
            meta = meta-meta
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)
        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):

            out, attn_dist, _, _,_ = self.decoder(self.embedding(ys)+meta.unsqueeze(1),encoder_outputs, False, (mask_src,mask_trg))
            
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])

            if config.USE_CUDA:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent

class CvaeNAD(nn.Module):

    def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(CvaeNAD, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        
        self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        if config.num_var_layers>0:
            self.decoder = VarDecoder2(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
        else:
            self.decoder = VarDecoder3(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):

        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'r_encoder_state_dict': self.r_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Response encode
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])
        r_encoder_outputs = self.r_encoder(self.embedding(batch["posterior_batch"]),mask_res)
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch),mask_src)
        meta = self.embedding(batch["program_label"])

        
        # Decode
        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        latent_dim = meta.size()[-1]
        meta = meta.repeat(1,dec_batch.size(1)).view(dec_batch.size(0),dec_batch.size(1),latent_dim)
        pre_logit, attn_dist, mean, log_var = self.decoder(meta,encoder_outputs,r_encoder_outputs, (mask_src,mask_res,mask_trg))
        if not train:
            pre_logit, attn_dist, _, _= self.decoder(meta,encoder_outputs, None, (mask_src,None,mask_trg))
        ## compute output dist
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        kld_loss = gaussian_kld(mean["posterior"], log_var["posterior"],mean["prior"], log_var["prior"])
        kld_loss = torch.mean(kld_loss)
        kl_weight = min(iter/config.full_kl_step, 1) if config.full_kl_step >0 else 1.0
        loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss
        if(train):
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item()


    def decoder_greedy(self, batch, max_dec_step=50):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
  
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        meta = self.embedding(batch["program_label"])
        encoder_outputs = self.encoder(self.embedding(enc_batch),mask_src)

        mask_trg = torch.ones((enc_batch.size(0),50))
        meta_size = meta.size()
        meta = meta.repeat(1,50).view(meta_size[0],50,meta_size[1])
        out, attn_dist, _, _= self.decoder(meta,encoder_outputs, None, (mask_src,None,mask_trg))
        prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
        _, batch_out = torch.max(prob, dim = 1)
        batch_out = batch_out.data.cpu().numpy()
        sentences = []
        for sent in batch_out:
            st = ''
            for w in sent:
                if w == config.EOS_idx: break
                else: st+= self.vocab.index2word[w] + ' '
            sentences.append(st)
        return sentences
        
    def decoder_greedy_po(self, batch, max_dec_step=50):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)

        ## Response encode
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])
        r_encoder_outputs = self.r_encoder(self.embedding(batch["posterior_batch"]),mask_res)
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch),mask_src)
        meta = self.embedding(batch["program_label"])

        encoder_outputs = self.encoder(self.embedding(enc_batch),mask_src)
        mask_trg = torch.ones((enc_batch.size(0),50))
        meta_size = meta.size()
        meta = meta.repeat(1,50).view(meta_size[0],50,meta_size[1])
        out, attn_dist, mean, log_var = self.decoder(meta,encoder_outputs,r_encoder_outputs, (mask_src,mask_res,mask_trg))
        prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
        _, batch_out = torch.max(prob, dim = 1)
        batch_out = batch_out.data.cpu().numpy()
        sentences = []
        for sent in batch_out:
            st = ''
            for w in sent:
                if w == config.EOS_idx: break
                else: st+= self.vocab.index2word[w] + ' '
            sentences.append(st)
        return sentences

# class CvaeNAD(nn.Module):

#     def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False):
#         super(CvaeNAD, self).__init__()
#         self.vocab = vocab
#         self.vocab_size = vocab.n_words

#         self.embedding = share_embedding(self.vocab,config.pretrain_emb)
#         self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
#                                 total_key_depth=config.depth, total_value_depth=config.depth,
#                                 filter_size=config.filter,universal=config.universal)
        
#         self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
#                                 total_key_depth=config.depth, total_value_depth=config.depth,
#                                 filter_size=config.filter,universal=config.universal)
        
#         self.decoder = Decoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
#                                     total_key_depth=config.depth,total_value_depth=config.depth,
#                                     filter_size=config.filter)

#         self.generator = Generator(config.hidden_dim, self.vocab_size)

#         if config.weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.generator.proj.weight = self.embedding.lut.weight

#         self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
 
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
#         if(config.noam):
#             self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#         if model_file_path is not None:
#             print("loading weights")
#             state = torch.load(model_file_path, map_location= lambda storage, location: storage)
#             self.encoder.load_state_dict(state['encoder_state_dict'])
#             self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
#             self.decoder.load_state_dict(state['decoder_state_dict'])
#             self.generator.load_state_dict(state['generator_dict'])
#             self.embedding.load_state_dict(state['embedding_dict'])
#             self.recognet.load_state_dict(state['recognet_state_dict'])
#             self.priornet.load_state_dict(state['priornet_state_dict']) 
#             if (load_optim):
#                 self.optimizer.load_state_dict(state['optimizer'])
#             self.eval()

#         self.model_dir = config.save_path
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#         self.best_path = ""

#     def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):

#         state = {
#             'iter': iter,
#             'encoder_state_dict': self.encoder.state_dict(),
#             'r_encoder_state_dict': self.r_encoder.state_dict(),
#             'decoder_state_dict': self.decoder.state_dict(),
#             'generator_dict': self.generator.state_dict(),
#             'recognet_state_dict': self.recognet.state_dict(),
#             'priornet_state_dict': self.priornet.state_dict(),
#             'embedding_dict': self.embedding.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'current_loss': running_avg_ppl
#         }
#         model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
#         self.best_path = model_save_path
#         torch.save(state, model_save_path)

#     def train_one_batch(self, batch, iter, train=True):
#         enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
#         dec_batch, _, _, _, _ = get_output_from_batch(batch)
        
#         if(config.noam):
#             self.optimizer.optimizer.zero_grad()
#         else:
#             self.optimizer.zero_grad()

#         ## Response encode
#         mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
#         posterior_mask = self.embedding(batch["posterior_mask"])
#         r_encoder_outputs = self.r_encoder(self.embedding(batch["posterior_batch"]),mask_res)
#         ## Encode
#         mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
#         emb_mask = self.embedding(batch["input_mask"])
#         encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)
#         #latent variable
#         q_r = r_encoder_outputs[:,0]
#         q_p = encoder_outputs[:,0]
#         z_p, mean_p, log_var_p = self.priornet(q_p)
#         z_r, mean_r, log_var_r = self.recognet(q_r)
#         z = z_r if train else z_p
        
#         # Decode
#         latent_dim = z.size()[-1]
#         z = z.repeat(1,dec_batch.size(1)).view(dec_batch.size(0),dec_batch.size(1),latent_dim)

#         mask_trg = None
#         pre_logit, attn_dist = self.decoder(z,encoder_outputs, (mask_src,mask_trg))
        
#         ## compute output dist
#         logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
#         #logit = F.log_softmax(logit,dim=-1) #fix the name later
#         ## loss: NNL if ptr else Cross entropy
#         loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
#         kld_loss = gaussian_kld(mean_r, log_var_r, mean_p, log_var_p)
#         kld_loss = torch.mean(kld_loss)
#         kl_weight = min(iter/config.full_kl_step, 1.0) if config.full_kl_step >0 else 1.0
#         loss = loss_rec + kl_weight*kld_loss
#         if(train):
#             loss.backward()
#             self.optimizer.step()

#         return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item()



#     def decoder_greedy(self, batch, max_dec_step=30):
#         enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        
#         ## Encode
#         mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
#         emb_mask = self.embedding(batch["input_mask"])
#         encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)
#         #latent variable
#         q_p = encoder_outputs[:,0]
#         z_p, mean_p, log_var_p = self.priornet(q_p)
#         z = z_p
#         z_size = z.size()
#         z = z.repeat(1,30).view(z_size[0], 30, z_size[1])

#         mask_trg = None
#         out, attn_dist = self.decoder(z,encoder_outputs, (mask_src,mask_trg))
#         prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
#         _, batch_out = torch.max(prob, dim = 1)
#         batch_out = batch_out.data.cpu().numpy()
#         sentences = []
#         for sent in batch_out:
#             st = ''
#             for w in sent:
#                 if w == config.EOS_idx: break
#                 else: st+= self.vocab.index2word[w] + ' '
#             sentences.append(st)
#         return sentences
