"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from constants import *

import math

class VanillaConv(nn.Module):

    def __init__(self, Y, kernel_size, conv_dim_factor):
        super(VanillaConv, self).__init__()
        #embed
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        xavier_uniform(self.embed.weight)
        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)
        
        #conv
        self.conv = nn.Conv1d(EMBEDDING_SIZE, conv_dim_factor*Y, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)
        self.conv_drop = nn.Dropout(p=DROPOUT_DENSE)

        #dense
        self.fc = nn.Linear(conv_dim_factor*Y, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/pool
#        x = self.conv_drop(self.conv(x))
        x = self.conv(x)
        x = F.tanh(F.max_pool1d(x, kernel_size=x.size()[2]))
        x = x.squeeze(dim=2)

        #dense
        x = F.tanh(self.fc(x))

        return x

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
