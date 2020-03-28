import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config
from utils.metric import rouge, moses_multi_bleu
from model.common_layer import share_embedding, get_input_from_batch, get_output_from_batch, sequence_mask, gaussian_kld, PositionwiseFeedForward
import random
from numpy import random
import math
import pickle
import numpy as np
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self,vocab_size, embedding=None):
        super(Encoder, self).__init__()
        if(embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_dim)
            init_wt_normal(self.embedding.weight)

        self.lstm = nn.GRU(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        if config.use_oov_emb:
            self.oov_emb_proj = nn.Linear(2 * self.hidden_size, config.emb_dim)


    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        h = h.contiguous()
        #max_h, _ = h.max(dim=1)


        batch_size = input.size(0) 
        if config.use_oov_emb:
            for i in range(batch_size):
                for  j in range(seq_lens[i]):
                    if input[i, j] == config.UNK_idx:
                        unk_emb = torch.zeros(2 * self.hidden_size)
                        if config.USE_CUDA:
                            unk_emb = unk_emb.cuda()
                        if j > 0:
                            unk_emb[:config.hidden_size] = h[i, j - 1, :config.hidden_size]
                        if j < seq_lens[i] - 1:
                            unk_emb[config.hidden_size:] = h[i, j + 1, config.hidden_size:]
                        embedded[i, j] = self.oov_emb_proj(unk_emb)

            packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
            output, hidden = self.lstm(packed)
            h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n

        return h, hidden

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, h, enc_padding_mask, coverage):
        b, t_k, n = list(h.size())
        h = h.view(-1, n)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask.cuda() # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.view(-1, t_k, n)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding = None ):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        if(embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_dim)
            init_wt_normal(self.embedding.weight)

        self.reduce_h = nn.Linear(config.hidden_dim * 3, config.hidden_dim*2) # reduce input to 2*hidden_size
        init_linear_wt(self.reduce_h)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.GRU(config.emb_dim, config.hidden_dim*2, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, training=True):
        if step==0:
            s_t_1 = self.reduce_h(s_t_1).unsqueeze(0)

        if not training and step == 0:
            s_t_hat = s_t_1.squeeze(0)
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        s_t_hat = s_t.squeeze(0)

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage)

        if training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, 2*config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage




class Latent(nn.Module):
    def __init__(self,is_eval):
        super(Latent, self).__init__()
        self.mean = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.var = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.mean_p = PositionwiseFeedForward(config.hidden_dim*4, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.var_p = PositionwiseFeedForward(config.hidden_dim*4, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.is_eval = is_eval

    def forward(self,x,x_p, train=True):
        mean = self.mean(x)
        log_var = self.var(x)
        eps = torch.randn(mean.size())
        std = torch.exp(0.5 * log_var)
        if config.USE_CUDA: eps = eps.cuda()
        z = eps * std + mean
        kld_loss = 0
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x),dim=-1))
            kld_loss = gaussian_kld(mean_p,log_var_p,mean,log_var)
            kld_loss = torch.mean(kld_loss)
        if train:
            std = torch.exp(0.5 * log_var_p)
            if config.USE_CUDA: eps = eps.cuda()
            z = eps * std + mean_p
        return kld_loss, z

class SoftmaxOutputLayer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(SoftmaxOutputLayer, self).__init__()
        self.proj_hidden = nn.Linear(d_model*3, d_model)
        self.proj = nn.Linear(d_model, vocab)
        init_linear_wt(self.proj)

    def forward(self, x):
        x = self.proj_hidden(x)
        logit = self.proj(x)

        return F.log_softmax(logit,dim=-1)

class SeqToSeq(nn.Module):

    def __init__(self, vocab, model_file_path=None, is_eval=False):
        super(SeqToSeq, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = None

        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        self.encoder = Encoder(self.vocab_size, self.embedding)
        self.encoder_r = Encoder(self.vocab_size, self.embedding)
        self.decoder = Decoder(self.vocab_size, self.embedding)
        self.bow = SoftmaxOutputLayer(config.hidden_dim,self.vocab_size)
        self.latent = Latent(is_eval)
        #reduce_state = ReduceState()
        # shared the embedding between encoder and decoder
        # decoder.embedding.weight = encoder.embedding.weight


        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.encoder_r.load_state_dict(state['encoder_r_state_dict'])
            self.latent.load_state_dict(state['latent'])
            #self.reduce_state.load_state_dict(state['reduce_state_dict'])

        if config.USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder_r = self.encoder_r.cuda()
            self.latent = self.latent.cuda()
            self.bow = self.bow.cuda()
            #reduce_state = reduce_state.cuda()
        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            self.encoder_r = self.encoder_r.eval()
            self.latent = self.latent.eval()
            #reduce_state = reduce_state.eval()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            #'reduce_state_dict': self.reduce_state.state_dict(),
            'encoder_r_state_dict': self.encoder_r.state_dict(),
            'latent': self.latent.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, n_iter, train=True):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # sort response for lstm
        r_len = np.array(batch["posterior_lengths"])
        r_sort = r_len.argsort()[::-1]
        r_len = r_len[r_sort].tolist()
        unsort = r_sort.argsort()

        _, encoder_hidden_r = self.encoder_r(batch["posterior_batch"][r_sort.tolist()], r_len)
        #encoder_hidden_r = encoder_hidden_r[unsort.tolist()] #unsort

        s_t_1 = encoder_hidden.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2) #b x hidden_dim*2
        s_t_1_r = encoder_hidden_r.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)[unsort.tolist()] #unsort #b x hidden_dim*2
        batch_size = enc_batch.size(0)

        #meta = self.embedding(batch["program_label"])
        kld_loss,z = self.latent(s_t_1,s_t_1_r,train=True)
        if config.model=="seq2seq":
            z = z-z
            kld_loss = torch.Tensor([0])
        
        s_t_1 = torch.cat((z,s_t_1),dim=-1)
        
        if config.model=="cvae":
            z_logit = self.bow(s_t_1) # [batch_size, vocab_size]
            z_logit = z_logit.unsqueeze(1).repeat(1,dec_batch.size(1),1)
            loss_aux = self.criterion(z_logit.contiguous().view(-1, z_logit.size(-1)), dec_batch.contiguous().view(-1))
        y_t_1 = torch.LongTensor([config.SOS_idx] * batch_size)
        
        if config.USE_CUDA:
            y_t_1 = y_t_1.cuda()
        step_losses = []
        for di in range(max_dec_len):
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                            coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask.cuda()
            step_losses.append(step_loss)
            y_t_1 = dec_batch[:, di]  # Teacher forcing

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var.float().cuda()
        loss_rec = torch.mean(batch_avg_loss)
        if config.model == "cvae":
            kl_weight = min(math.tanh(6 * n_iter/config.full_kl_step - 3) + 1, 1)
            #kl_weight = min(n_iter/config.full_kl_step, 0.5) if config.full_kl_step >0 else 1.0
            loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + loss_aux* config.aux_ceiling
            elbo = loss_rec+kld_loss
        else:
            loss = loss_rec
            loss_aux = torch.Tensor([0])
            elbo = loss_rec
        if(train):
            loss.backward()

            self.optimizer.step()
        return loss_rec.item(), math.exp(loss_rec.item()), kld_loss.item(), loss_aux.item(), elbo.item()


    def decoder_greedy(self, batch, max_dec_step=31):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = get_input_from_batch(batch)

        encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        
        s_t_1 = encoder_hidden.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2) #b x hidden_dim*2

        kld_loss,z = self.latent(s_t_1,None,False)
        if config.model=="seq2seq":
            z = z-z
        
        s_t_1 = torch.cat((z,s_t_1),dim=-1)

        batch_size = enc_batch.size(0)
        y_t_1 = torch.LongTensor([config.SOS_idx] * batch_size)
        if config.USE_CUDA:
            y_t_1 = y_t_1.cuda()

        decoded_words = []
        for di in range(max_dec_step):
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage = self.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                        coverage, di)

            _, topk_ids = torch.topk(final_dist,1)
            decoded_words.append(['<EOS>'if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in topk_ids.view(-1)])
            
            if config.USE_CUDA:
                y_t_1 = topk_ids.squeeze(-1).cuda() # Teacher forcing
            else:
                y_t_1 = topk_ids.squeeze(-1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent



