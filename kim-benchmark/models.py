"""
    Holds PyTorch models

    Copies initialization choices from Denny Britz's implementation in TensorFlow (https://github.com/dennybritz/cnn-text-classification-tf)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from constants import *

import math

class MultiConv(nn.Module):
    
    def __init__(self, W, min_filter, max_filter, num_filter_maps, s):
        super(MultiConv, self).__init__()
        self.s = s
        #embed
        #init with word2vec pretrained embeddings (making it part of computation graph for non-static impl)
        self.embed = nn.Embedding(W.size()[0], W.size()[1])
        self.embed.weight.data = W.clone()
        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)

        self.convs = []
        for idx,filter_size in enumerate(range(min_filter, max_filter+1)):
            conv = nn.Conv2d(1, num_filter_maps, (filter_size, W.size()[1]) )
            conv.weight.data.uniform_(-0.01, 0.01)
            conv.bias.data.fill_(0)
            self.convs.append(conv)
            self.add_module("conv-%d" % (filter_size), conv)
        self.conv_drop = nn.Dropout(p=DROPOUT_DENSE)

        self.fc = nn.Linear(num_filter_maps*(max_filter-min_filter+1), 2)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)
        self.fc_drop = nn.Dropout(p=DROPOUT_DENSE)

    def forward(self, x):
        #embed
        x = self.embed(x)
        x = x.unsqueeze(dim=1)

        #conv/pool/concat
        pool_outputs = []
        for i in range(len(self.convs)):
            conv = self.convs[i]
            conv_output = F.relu(conv(x)).squeeze(dim=3)
            pool_output = F.max_pool1d(conv_output, kernel_size=conv_output.size()[2])
            pool_output = pool_output.squeeze(dim=2)
            pool_outputs.append(pool_output)
        x = torch.cat(pool_outputs, 1)

        #dense
        x = self.fc(self.fc_drop(x))

        #activation
        return F.softmax(x)

    def enforce_norm_constraint(self):
        self.fc.weight.data.renorm(p=2, dim=0, maxnorm=self.s)
        for conv in self.convs:
            conv.weight.data.renorm(p=2, dim=0, maxnorm=self.s)

class VanillaConv(nn.Module):

    def __init__(self, W, kernel_size, num_filter_maps, s):
        super(VanillaConv, self).__init__()
        self.s = s
        #embed
        #init with word2vec pretrained embeddings (making it part of computation graph for non-static impl)
        self.embed = nn.Embedding(W.size()[0], W.size()[1])
        self.embed.weight.data = W.clone()
        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)
        
        #conv
        self.conv = nn.Conv1d(W.size()[1], num_filter_maps, kernel_size=kernel_size)
        self.conv.weight.data.uniform_(-0.01, 0.01)
        self.conv.bias.data.fill_(0)
        self.conv_drop = nn.Dropout(p=DROPOUT_DENSE)

        #dense
        self.fc = nn.Linear(num_filter_maps, 2)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)
        self.fc_drop = nn.Dropout(p=DROPOUT_DENSE)

    def forward(self, x):
        #embed
        x = self.embed(x)
        x = x.transpose(1, 2)

        #conv/pool
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, kernel_size=x.size()[2])
        x = x.squeeze(dim=2)

        #dense, linear activation
        x = self.fc(self.conv_drop(x))

        #activation
        return F.softmax(x)
   
    def enforce_norm_constraint(self):
        self.fc.weight.data.renorm(p=2, dim=0, maxnorm=self.s)
        self.conv.weight.data.renorm(p=2, dim=0, maxnorm=self.s)

#copied from current pytorch source which I couldn't build
def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


#copied from current pytorch source which I couldn't build
def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty
       of training deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.

       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in + fan_out)) * sqrt(3)
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_uniform(w, gain=math.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return tensor.uniform_(-a, a)

