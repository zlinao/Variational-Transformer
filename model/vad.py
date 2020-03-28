# from https://github.com/likecoffee/VariationalAutoRegressive
#import ipdb
import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import torch.nn.functional as F
import math

from model import rnn_cell
#import rnn_cell

class Configuration(object):
    def __init__(self, logging_file_name, learning_dict ,common_dict, encoder_dict, decoder_dict, interval_dict):
        self._logging_file_name = logging_file_name
        self._learning_dict = learning_dict
        self._interval_dict = interval_dict
        self._common_dict = common_dict
        self._encoder_dict = encoder_dict
        self._decoder_dict = decoder_dict

    @classmethod
    def load(cls):
        data_dict = {
            "logging_file_name":'null',
            "encoder":{
                "rnn_size": 1024,
                "rnn_num_layer": 2,
                "bidirectional": 'true'
            },
            "decoder":{
                "rnn_size": 800,
                "rnn_num_layer": 2,
                "attn_type": "last",
                "layer_normed": 'false',
                "additional_rnn_size":512,
                "mlp_size":512,
                "z_size":300,
                "dropout":0.3
            },
            "common":{
                "num_word": 54769,
                "sample_number":1,
                "emb_size": 300,
                "context_size": 512,
                "beam_size":3
            },
            "interval":{
                "report": 1000,
                "evaluation": 6000,
                "save" : 6000,
                "generation" : 6000
            },
            "learning":{
                    "parent_name": "restore",
                    "batch_size":40,
                    "lr":1e-4,
                    "clip_norm":2,
                    "num_epoch":200,
                    "cuda":3
            }
        }
        # with open(json_file_name) as f:
        #     data_dict = json.load(f)
        return cls(data_dict['logging_file_name'],data_dict['learning'],data_dict['common'],data_dict['encoder'],data_dict['decoder'],data_dict['interval'])

    # def save(self, json_file_name):
    #     data_dict = {"common":self._common_dict,"encoder":self._encoder_dict,"decoder":self._decoder_dict,
    #                  "learning":self._learning_dict,"interval":self._interval_dict}
    #     with open(json_file_name,"w") as wf:
    #         json.dump(data_dict, wf)

    def __repr__(self):
        learning_s= "Learning: \n" + "\n".join(["\t{0} : {1}".format(name,self._learning_dict[name]) for name in self._learning_dict.keys()])+"\n"
        common_s =  "Common:  \n" + "\n".join(["\t{0} : {1}".format(name,self._common_dict[name]) for name in self._common_dict.keys()])+"\n"
        encoder_s = "Encoder: \n" + "\n".join(["\t{0} : {1}".format(name,self._encoder_dict[name]) for name in self._encoder_dict.keys()])+"\n"
        decoder_s = "Decoder: \n" + "\n".join(["\t{0} : {1}".format(name,self._decoder_dict[name]) for name in self._decoder_dict.keys()])+"\n"

        return learning_s+common_s+encoder_s+decoder_s

    @property
    def common(self):
        return self._common_dict
    @property
    def encoder(self):
        encoder_config_dict = self._encoder_dict
        encoder_config_dict.update(self._common_dict)
        return encoder_config_dict
    @property
    def decoder(self):
        decoder_config_dict = self._decoder_dict
        decoder_config_dict.update(self._common_dict)
        return decoder_config_dict
    @property
    def learning(self):
        return self._learning_dict
    @property
    def interval(self):
        return self._interval_dict
    @property
    def logging_file_name(self):
        return self._logging_file_name



class Attention(nn.Module):
    def __init__(self, context_dim, hidden_dim, type="mlp"):
        super(Attention,self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.type = type
        if type== "mlp":
            self.attn = nn.Linear(self.hidden_dim + self.context_dim, hidden_dim)
            self.v = nn.Parameter(torch.rand(hidden_dim))
        elif type == "last":
            None
        elif type == "mean":
            None
        elif type == "general" or type == "dot":
            raise NotImplementedError("General and Dot not implemented")
        else:
            raise  Exception("Wrong Atten Type")
        self.init_weight()

    def __repr__(self):
        s = "type = {}, context_dim= {}, hidden_dim= {}".format(self.type, self.context_dim, self.hidden_dim)
        return s
    def init_weight(self):
        if self.type == "mlp":
            nn.init.xavier_normal_(self.attn.weight)
            nn.init.uniform_(self.attn.bias,-0.1,0.1)
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)
        elif self.type == "general":
            raise NotImplementedError("General and Dot not implemented")
        else:
            None
    def score(self, hidden, context):
        attn_input= torch.cat([hidden,context],dim=2)
        energy = F.tanh(self.attn(attn_input))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(context.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

    def forward(self, hidden, context):
        src_seq_len = context.size(0)

        if self.type == 'general' or self.type == 'dot':
            raise NotImplementedError("General and Dot not implemented")
        elif self.type == "last":
            attn_score = 1
            attn_context = torch.stack([context[0],context[-1]],dim=0)
            attn_context = torch.mean(attn_context,dim=0)
        elif self.type == "mean":
            attn_score = 1
            attn_context = torch.mean(context,dim=0)
        elif self.type == "mlp":
            H = hidden.unsqueeze(0).repeat(src_seq_len, 1, 1).transpose(0, 1)  # [B*T*H]
            context = context.transpose(0, 1)  # [B*T*H]
            attn_energies = self.score(H, context)  # compute attention score
            attn_score = nn.functional.softmax(attn_energies, dim=1)  # normalize with softmax [B*1*T]
            attn_context = attn_score.unsqueeze(1).bmm(context)
            attn_context = attn_context.squeeze(1)  # B*H

        return attn_score,attn_context

def gaussian_kld(mu_1, logvar_1, mu_2, logvar_2, mean=False):
    loss = (logvar_2 - logvar_1) + (torch.exp(logvar_1) / torch.exp(logvar_2)) + ((mu_1 - mu_2) ** 2 / torch.exp(logvar_2) - 1.)
    loss = loss / 2
    if mean:
        loss = torch.mean(loss, dim=1)
    else:
        loss = torch.sum(loss, dim=1)
    stochastic_list = [mu_1.detach().cpu().numpy(), logvar_1.detach().cpu().numpy(), mu_2.detach().cpu().numpy(), logvar_2.detach().cpu().numpy()]
    return loss, stochastic_list


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, addtional_input_size, mlp_size, z_size):
        self.input_size = input_size
        self.mlp_size = mlp_size
        self.additional_input_size = addtional_input_size
        self.z_size = z_size
        super(VariationalAutoEncoder, self).__init__()
        self.inference_linear = nn.Sequential(
            nn.Linear(input_size + addtional_input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )
        self.prior_linear = nn.Sequential(
            nn.Linear(input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )

    @staticmethod
    def reparameter(mu, logvar, random_variable=None):
        std = logvar.mul(0.5).exp_()
        if random_variable is None:
            random_variable = mu.new(*mu.size()).normal_()
            return random_variable.mul(std).add_(mu)
        else:
            if len(random_variable.size()) == 3:
                sampled_random_variable = random_variable.mul(std.unsqueeze(0)).add_(mu.unsqueeze(0))
                return random_variable
            elif len(random_variable.size()) == 2:
                return random_variable.mul(std).add_(mu)
            else:
                raise Exception("Wrong size of given random variable")

    def forward(self, input, additional_input=None, random_variable=None, inference_mode=True):
        prior_gaussian_paramter = self.prior_linear(input)
        prior_gaussian_paramter = torch.clamp(prior_gaussian_paramter, -4, 4)
        prior_mu, prior_logvar = torch.chunk(prior_gaussian_paramter, 2, 1)
        if inference_mode:
            assert additional_input is not None
            inference_input = torch.cat([input, additional_input], dim=1)
            inference_gaussian_paramter = self.inference_linear(inference_input)
            inference_gaussian_paramter = torch.clamp(inference_gaussian_paramter, -4, 4)
            inference_mu, inference_logvar = torch.chunk(inference_gaussian_paramter, 2, 1)
            z = VariationalAutoEncoder.reparameter(inference_mu, inference_logvar, random_variable)
            kld,stochastic_var_list = gaussian_kld(inference_mu, inference_logvar, prior_mu, prior_logvar)
            return z, kld, stochastic_var_list
        else:
            z = VariationalAutoEncoder.reparameter(
                prior_mu, prior_logvar, random_variable)
            return z


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, rnn_size, rnn_num_layer,
                 context_size, bidirectional, **argv):
        self.input_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_num_layer = rnn_num_layer
        self.bidirectional = bidirectional
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=rnn_size,
            num_layers=rnn_num_layer,
            bidirectional=bidirectional)
        self.context_linear = nn.Linear(
            rnn_size * (2 if bidirectional else 1),
            context_size)

    def init_rnn_hidden(self, batch_size):
        param_data = next(self.parameters())
        bidirectional_multipier = 2 if self.bidirectional else 1
        rnn_whole_hidden = param_data.new(self.rnn_num_layer*bidirectional_multipier,batch_size,self.rnn_size).zero_()

        return rnn_whole_hidden

    def forward(self, input, length):
        seq_len, batch_size, _ = input.size()
        hidden = self.init_rnn_hidden(batch_size)
        packed_input = nn.utils.rnn.pack_padded_sequence(input, length)
        rnn_output, hidden = self.rnn(packed_input, hidden)
        rnn_output_padded, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)
        output = self.context_linear(rnn_output_padded)
        return output


class VariationalAutoregressiveDecoder(nn.Module):
    def __init__(
            self, num_word, emb_size, context_size, rnn_size, rnn_num_layer,
            layer_normed, mlp_size, z_size, attn_type, dropout, **argv):
        self.num_word = num_word
        self.emb_size = emb_size
        self.context_size = context_size
        self.rnn_size = rnn_size
        self.rnn_num_layer = rnn_num_layer
        self.layer_normed = layer_normed
        self.mlp_size = mlp_size
        self.z_size = z_size
        self.dropout = dropout
        super(VariationalAutoregressiveDecoder, self).__init__()
        if layer_normed:
            self.rnn = rnn_cell.StackedLayerNormedGRUCell(
                emb_size + context_size+z_size, rnn_size, rnn_num_layer, 0)
        else:
            self.rnn = rnn_cell.StackedGRUCell(
                emb_size + context_size+z_size, rnn_size, rnn_num_layer, 0)
        self.output_linear = nn.Linear(rnn_size + context_size+z_size, num_word)
        if attn_type is not None:
            self.attention = Attention(context_size, rnn_size, attn_type)
        self.bwd_rnn = nn.GRU(emb_size, rnn_size, num_layers=rnn_num_layer)
        self.bwd_output_linear = nn.Linear(rnn_size, num_word)
        self.bwd_rnn_vae = VariationalAutoEncoder(
            rnn_size + context_size, rnn_size, mlp_size, z_size)
        self.bow_linear = nn.Sequential(nn.Linear(rnn_size + z_size, num_word))
        self.output_linear = nn.Linear(rnn_size + context_size + z_size, num_word)
        self.context_to_hidden = nn.Linear(
            context_size, rnn_num_layer * rnn_size)

    def init_rnn_hidden(self, context=None, batch_size=None):
        if context is not None:
            batch_size = context.size(1)
            mean_context = torch.mean(context, dim=0)
            rnn_whole_hidden = self.context_to_hidden(mean_context)
            rnn_whole_hidden = rnn_whole_hidden.reshape(batch_size, self.rnn_num_layer, self.rnn_size)
            rnn_whole_hidden = rnn_whole_hidden.permute(1, 0, 2).contiguous()
            return rnn_whole_hidden
        elif batch_size is not None:
            param_data = next(self.parameters())
            rnn_whole_hidden = param_data.new(self.rnn_num_layer, batch_size, self.rnn_size).zero_()

            return rnn_whole_hidden

    def bwd_pass(self, target_emb_input, context, target, target_mask):
        seq_length = target_emb_input.size(0)
        batch_size = target_emb_input.size(1)
        target_emb_input = target_emb_input.detach()
        idx = torch.LongTensor(np.arange(target_emb_input.size(0))[::-1].tolist())
        idx = target.new(seq_length).copy_(idx)
        emb_input_bwd = target_emb_input.index_select(0, idx)
        reverse_target = target.index_select(0, idx)
        reverse_zero_pad = reverse_target.new(1, batch_size).zero_()
        reverse_target = torch.cat([reverse_target[1:], reverse_zero_pad], dim=0)
        reverse_target_mask = target_mask.index_select(0, idx)
        reverse_target_mask_zero_pad = reverse_target_mask.new(1, batch_size).zero_()
        reverse_target_mask = torch.cat([reverse_target_mask[1:], reverse_target_mask_zero_pad], dim=0)
        hidden = self.init_rnn_hidden(context)
        states, _ = self.bwd_rnn(emb_input_bwd, hidden)
        bwd_output = self.bwd_output_linear(states)
        states = states.index_select(0, idx)
        reverse_ce = nn.functional.cross_entropy(bwd_output.view(bwd_output.size(
            0) * bwd_output.size(1), -1), reverse_target.view(-1), reduce=False)
        #print(reverse_ce.shape)
        #print(reverse_target_mask.view(seq_length *batch_size).shape)
        reverse_ce = torch.mean(reverse_ce *reverse_target_mask.view(seq_length *batch_size))
        return states, reverse_ce

    def train_forward(self, emb_input, context, target,target_emb_input,target_mask,sample_number):
        ce_list, bwd_rnn_kld_list, aux_bow_loss_list = [], [], []
        whole_stochastic_list = []
        seq_length, batch_size, emb_size = emb_input.size()
        bwd_states, bwd_ce = self.bwd_pass(target_emb_input, context, target, target_mask)
        if  self.training and sample_number > 1:
            emb_input = emb_input.repeat(1, sample_number, 1)
            target = target.repeat(1, sample_number)
            random_variable_bwd_rnn = emb_input.new(seq_length, batch_size*sample_number, self.z_size).normal_()
            bwd_states = bwd_states.repeat(1, sample_number, 1)
            target_mask = target_mask.repeat(1, sample_number)
        else:
            random_variable_bwd_rnn = emb_input.new(seq_length, batch_size*sample_number, self.z_size).normal_()

        if self.training:
            dropout_mask_data = emb_input.new(seq_length, batch_size * sample_number, 1).fill_(1 - self.dropout)
            dropout_mask_data = torch.bernoulli(dropout_mask_data)
            dropout_mask_data = dropout_mask_data.repeat(1, 1, emb_size)
            dropout_mask = dropout_mask_data
            emb_input = emb_input * dropout_mask
        rnn_whole_hidden = self.init_rnn_hidden(context=context)
        context = context.repeat(1, sample_number, 1)
        rnn_whole_hidden = rnn_whole_hidden.repeat(1, sample_number, 1)
        rnn_last_hidden = rnn_whole_hidden[-1]
        
        for step_i in range(seq_length):
            # get step variable
            emb_input_step = emb_input[step_i]
            score, attn_context_step = self.attention(rnn_last_hidden, context)
            bwd_state_step = bwd_states[step_i]
            random_variable_bwd_rnn_step = random_variable_bwd_rnn[step_i]
            target_mask_step = target_mask[step_i]
            # VAE process
            if self.training:
                z_bwd_rnn, kld_bwd_rnn,stochastic_list = self.bwd_rnn_vae.forward(
                    input=torch.cat([rnn_last_hidden, attn_context_step],dim=1),
                    additional_input=bwd_state_step,
                    random_variable=random_variable_bwd_rnn_step,
                    inference_mode=True)
                whole_stochastic_list.append(stochastic_list)
            else:
                z_bwd_rnn = self.bwd_rnn_vae.forward(input = torch.cat([rnn_last_hidden, attn_context_step],dim=1), 
                random_variable = random_variable_bwd_rnn_step, 
                inference_mode=False)
            # RNN process
            rnn_input = torch.cat([emb_input_step, attn_context_step, z_bwd_rnn],dim=1)
            rnn_last_hidden, rnn_whole_hidden = self.rnn(rnn_input, rnn_whole_hidden)
            output_input = torch.cat([rnn_last_hidden, attn_context_step, z_bwd_rnn], dim=1)
            output = self.output_linear(output_input)
            ce = nn.functional.cross_entropy(output, target[step_i], reduce=False)
            ce = ce * target_mask_step
            # BOW Auxiliary Process
            if self.training:
                if seq_length - step_i > 5:
                    bow_truncated = step_i + 5
                else:
                    bow_truncated = seq_length
                bow_predicted_input = torch.cat([rnn_last_hidden, z_bwd_rnn], dim=1)
                bow_predicted = self.bow_linear(bow_predicted_input)
                bow_predicted = bow_predicted.repeat(bow_truncated - step_i, 1)
                bow_target = target[step_i:bow_truncated, :]
                bow_target = bow_target.reshape((bow_truncated - step_i) * batch_size * sample_number)
                aux_bow_loss = nn.functional.cross_entropy(bow_predicted, 
                        bow_target.view(batch_size *sample_number *(bow_truncated - step_i)), 
                        reduce=False)
                aux_bow_loss = aux_bow_loss.reshape(-1, (bow_truncated-step_i))
                aux_bow_loss = aux_bow_loss.mean(1)
            # IWAE weighted sum
            if self.training  and sample_number > 1:
                ce = ce.reshape(batch_size, sample_number)
                ce = torch.mean(ce, dim=1)
                kld_bwd_rnn = kld_bwd_rnn.reshape(batch_size, sample_number)
                kld_bwd_rnn = torch.mean(kld_bwd_rnn, dim=1)
                aux_bow_loss = aux_bow_loss.reshape(batch_size, sample_number)
                aux_bow_loss = torch.mean(aux_bow_loss, dim=1)
            # Collect Loss
            ce_list.append(ce)
            if self.training:
                bwd_rnn_kld_list.append(kld_bwd_rnn)
                aux_bow_loss_list.append(aux_bow_loss)
        
        ce_mean = torch.stack(ce_list, dim=0).mean()
        if not self.training:
            loss_dict = dict(ce = ce_mean)
        else:
            bwd_rnn_kld_mean = torch.stack(bwd_rnn_kld_list, dim=0).mean()
            aux_bow_loss_mean = torch.stack(aux_bow_loss_list, dim=0).mean()
            loss_dict = dict(ce=ce_mean,bwd_rnn_kld=bwd_rnn_kld_mean,
                aux_bow=aux_bow_loss_mean,bwd_ce=bwd_ce)
            stochastic_array_list = [np.stack(item,axis=0) for item in whole_stochastic_list]
            stochastic_whole_array = np.stack(stochastic_array_list,axis=0)
            stochastic_whole_array = stochastic_whole_array.transpose(1,2,0,3)
        
        if self.training:
            return loss_dict, stochastic_whole_array
        else:
            return loss_dict

    def generate_forward_step(self, emb_input_step,
                              context, rnn_whole_hidden=None):
        if rnn_whole_hidden is None:
            rnn_whole_hidden = self.init_rnn_hidden(context=context)
        rnn_last_hidden = rnn_whole_hidden[-1]
        score, attn_context_step = self.attention(rnn_last_hidden, context)
        vae_input = torch.cat([rnn_last_hidden, attn_context_step], dim=1)
        z_bwd_rnn = self.bwd_rnn_vae.forward(vae_input, inference_mode=False)
        rnn_input = torch.cat(
            [emb_input_step, attn_context_step, z_bwd_rnn], dim=1)
        rnn_last_hidden, rnn_whole_hidden = self.rnn(
            rnn_input, rnn_whole_hidden)
        output_input = torch.cat([rnn_last_hidden, attn_context_step, z_bwd_rnn], dim=1)
        output = self.output_linear(output_input)
        return output, rnn_whole_hidden


class Seq2SeqVAD(nn.Module):
    def __init__(self):
        super(Seq2SeqVAD, self).__init__()
        config = Configuration.load()
        common = config.common
        self.sample_number = common['sample_number']
        config.encoder.update(common)
        config.decoder.update(common)
        self.embedding = nn.Embedding(
            num_embeddings=common['num_word'],
            embedding_dim=common['emb_size'])
        self.encoder = RNNEncoder(**config.encoder)
        self.decoder = VariationalAutoregressiveDecoder(**config.decoder)

    def init_weight(self, pretrained_embedding=None):
        if pretrained_embedding is not None:
            if isinstance(self.embedding, nn.Sequential):
                self.embedding[0].weight.data = self.embedding[0].weight.data.new(pretrained_embedding)
            else:
                self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)
        else:
            if isinstance(self.embedding, nn.Sequential):
                self.embedding[0].weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    def pretrain_decoder_backward_rnn(self, src, tgt, src_len, tgt_mask):
        src_seq_len, batch_size = src.size()
        tgt_seq_len, tgt_batch_size = tgt.size()
        assert batch_size == tgt_batch_size

        src_emb = self.embedding(src)
        tgt_prefixed_wit_zero = torch.cat([tgt.new(1, batch_size).zero_(),tgt],dim=0)  # (tgt_seq+1) * batch_size
        # (tgt_seq_len+1) * batch_size * emb_size
        tgt_emb = self.embedding(tgt_prefixed_wit_zero)
        # tgt_seq_len * batch_size * emb_size (from index 0 to index n-1)
        tgt_emb_target = tgt_emb[1:]
        # src_seq_len * batch_size * context_size
        context = self.encoder(src_emb, src_len)

        _,reverse_ce = self.decoder.bwd_pass(tgt_emb_target, context, tgt, tgt_mask)

        return reverse_ce

    def train_forward(self, src, tgt, src_len, tgt_mask):
        src_seq_len, batch_size = src.size()
        tgt_seq_len, tgt_batch_size = tgt.size()
        assert batch_size == tgt_batch_size

        src_emb = self.embedding(src)
        tgt_prefixed_wit_zero = torch.cat(
            [tgt.new(1, batch_size).zero_(),tgt],dim=0)  # (tgt_seq+1) * batch_size
        # (tgt_seq_len+1) * batch_size * emb_size
        tgt_emb = self.embedding(tgt_prefixed_wit_zero)
        # tgt_seq_len * batch_size * emb_size (from index 0 to index n-1)
        tgt_emb_input = tgt_emb[:-1]
        # # tgt_seq_len * batch_size * emb_size (from index 1 to index n)
        tgt_emb_target = tgt_emb[1:]
        # src_seq_len * batch_size * context_size
        context = self.encoder(src_emb, src_len)

        loss_dict = self.decoder.train_forward(
            tgt_emb_input, context, tgt, tgt_emb_target, tgt_mask, self.sample_number)

        return loss_dict

    def generate_encoder_forward(self, src, src_len):
        src_emb = self.embedding(src)
        context = self.encoder(src_emb, src_len)

        return context

    def generate_decoder_forward(
            self, context, tgt_step=None, rnn_whole_hidden=None):
        if tgt_step is None:
            batch_size = context.size(1)
            tgt_step = context.new(batch_size).long().zero_()
        tgt_emb_step = self.embedding(tgt_step)
        output, rnn_whole_hidden = self.decoder.generate_forward_step(tgt_emb_step,context,rnn_whole_hidden)

        return output, rnn_whole_hidden