import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        assert  bias == False
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gain_ih = nn.Parameter(torch.ones(1, self.hidden_size*3))
        self.bias_ih = nn.Parameter(torch.zeros(1, self.hidden_size*3))
        self.gain_hh = nn.Parameter(torch.ones(1, self.hidden_size*3))
        self.bias_hh = nn.Parameter(torch.zeros(1, self.hidden_size*3))
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size,3*hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size,3*hidden_size))

        self.eps = 1e-5
        self.init_weight()

    def _layernorm(self,input,gain,bias):
        mean = input.mean(1, keepdim=True).expand_as(input)
        std = torch.sqrt(torch.var(input, dim=-1, keepdim=True) + self.eps)
        x = (input - mean) / std

        return x * gain.expand_as(x) + bias.expand_as(x)

    def init_weight(self):
        nn.init.xavier_normal(self.weight_ih.data)
        nn.init.xavier_normal(self.weight_hh.data)

    def forward(self, input, hx):
        gi = torch.mm(input,self.weight_ih)
        gh = torch.mm(hx,self.weight_hh)
        gi = self._layernorm(gi,self.gain_ih,self.bias_ih)
        gh = self._layernorm(gh,self.gain_hh,self.bias_hh)
        result = fusedBackend.GRUFused.apply(gi,gh,hx)

        return result
    def __repr__(self):
        string = "{}({},{})".format(self.__class__.__name__,self.input_size,self.hidden_size)
        return string

class StackedGRUCell(nn.Module):
    def __init__(self, input_size, rnn_size, num_layer, dropout):
        super(StackedGRUCell, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layer = num_layer
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(nn.GRUCell(input_size,rnn_size))
            input_size = rnn_size
        self.init_weight()

    def init_weight(self):
        for name,param in self.named_parameters():
            if "bias" in name:
                nn.init.uniform(param,-0.1,0.1)
            elif "weight" in name:
                nn.init.xavier_normal(param)
            else:
                raise ValueError("Wrong parameter")
    def forward(self, input, previous_hidden):
        current_hidden_state = []
        for i, layer in enumerate(self.layers):
            hidden_i = layer.forward(input, previous_hidden[i])
            input = hidden_i
            current_hidden_state += [hidden_i]
            if i + 1 != self.num_layer and self.training:
                input = self.dropout(input)
        current_hidden_state = torch.stack(current_hidden_state)
        output = input
        return output, current_hidden_state

    def __repr__(self):
        string = "{}(\n".format(self.__class__.__name__)
        for i, layer in enumerate(self.layers):
            layer_string = "({}): {} \n".format(i, repr(layer))
            string = string + layer_string
        string = string + ")\n"
        return string

class StackedLayerNormedGRUCell(nn.Module):
    def __init__(self, input_size, rnn_size, num_layer, dropout):
        super(StackedLayerNormedGRUCell, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layer = num_layer
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            if i == num_layer - 1:
                self.layers.append(LayerNormGRUCell(input_size,rnn_size))
            else:
                self.layers.append(nn.GRUCell(input_size,rnn_size))
            input_size = rnn_size
    def init_weight(self):
        for i in range(self.num_layer):
            if i == self.num_layer - 1:
                self.layers[i].init_weight()
            else:
                layer = self.layers[i]
                nn.init.xavier_normal(layer.weight_hh)
                nn.init.xavier_normal(layer.weight_ih)
    def forward(self, input, previous_hidden):
        current_hidden_state = []
        for i, layer in enumerate(self.layers):
            hidden_i = layer.forward(input, previous_hidden[i])
            input = hidden_i
            current_hidden_state += [hidden_i]
            if i + 1 != self.num_layer and self.training:
                input = self.dropout(input)
        current_hidden_state = torch.stack(current_hidden_state)
        output = input
        return output, current_hidden_state

    def __repr__(self):
        string = "{}(\n".format(self.__class__.__name__)
        for i,layer in enumerate(self.layers):
            layer_string = "\t({}): {} \n".format(i,repr(layer))
            string = string + layer_string
        string = string + ")\n"
        return string