"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from constants import *
from dataproc import extract_wvs

import math
import random
import sys

class Combiner(nn.Module):
    def __init__(self, Y, min_filter, max_filter, num_filter_maps):
        super(Combiner, self).__init__()
        self.convnet = MultiConv(Y, min_filter, max_filter, num_filter_maps)
        self.codes_mlp = CodesMLP(Y, MLP_OUTPUT)
        
        combined_dim = num_filter_maps*(max_filter - min_filter + 1) + MLP_OUTPUT
        self.combiner = nn.Linear(combined_dim, Y)

    def forward(self, x, start_inds=None):
        #x is a tuple: (Variable cur_codes, Variable cur_notes)
        x = torch.cat([self.convnet(x[1], start_inds).squeeze(2), F.relu(self.codes_mlp(x[0]))], 1)
        return F.relu(self.combiner(x))

class MultiConv(nn.Module):
    
    def __init__(self, Y, min_filter, max_filter, num_filter_maps):
        super(MultiConv, self).__init__()
        #embed
        #init with word2vec pretrained embeddings (making it part of computation graph for non-static impl)
        #load pretrained embeddings
        print("loading word embeddings...")
        W = torch.Tensor(extract_wvs.load_embeddings(Y, "processed"))

        #normalize them
        sizes = W.mul(W).sum(dim=1)
        #exclude first row, which is pad token
        #more pytorch-y way to do this?
        for i in range(1, W.size()[0]):
            W[i] = W[i].mul(1./math.sqrt(sizes[i][0]))

        self.embed = nn.Embedding(W.size()[0], W.size()[1])
#        self.embed = nn.Embedding(99200, EMBEDDING_SIZE)
        self.embed.weight.data = W.clone()
        #xavier_uniform(self.embed.weight)
        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)

        self.convs = []
        for idx,filter_size in enumerate(range(min_filter, max_filter+1)):
#        for filter_size in [3, 6, 9]:
            conv = nn.Conv2d(1, num_filter_maps, (filter_size, EMBEDDING_SIZE) )
            conv.weight.data.uniform_(-0.01, 0.01)
            conv.bias.data.fill_(0)
            self.convs.append(conv)
            self.add_module("conv-%d" % (filter_size), conv)
        self.fc_drop = nn.Dropout(p=DROPOUT_DENSE)

    def forward(self, x, start_inds=None):
        #embed
        x = self.embed(x)
#        x = self.embed_drop(x)
        x = x.unsqueeze(dim=1)

        #conv/pool/concat
        pool_outputs = []
        if start_inds is not None:
            inds = start_inds.copy().tolist()
            inds.append(sys.maxint)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            conv_output = F.relu(conv(x)).squeeze(dim=3)
            pool_output = F.max_pool1d(conv_output, kernel_size=conv_output.size()[2])

            #fancy batching stuff 
            #split batch into original docs
            if start_inds is not None:
                xs = pool_output.split(1)
                doc_pools = [ torch.cat(xs[inds[i]:inds[i+1]],2)
                                       for i in range(len(inds)-1) ]
                #then pool again to get max over maxes for each doc split
                final_pools = [F.relu(F.max_pool1d(doc_pools[i],
                                kernel_size=doc_pools[i].size()[2]))
                                for i in range(len(doc_pools))]
                #now that tensors are same size, cat them back together
                final_pool = torch.cat(final_pools).squeeze(dim=2)
                pool_outputs.append(final_pool)
            else:
                pool_outputs.append(pool_output)
        x = torch.cat(pool_outputs, 1)

        #dense
        x = self.fc_drop(x)
        return x

class CodesMLP(nn.Module):

    def __init__(self, Y, mlp_output):
        super(CodesMLP, self).__init__()
        #embed
        self.embed = nn.Embedding(Y, CODE_EMBEDDING_SIZE)
        xavier_uniform(self.embed.weight)

        self.fc1 = nn.Linear(CODE_EMBEDDING_SIZE, CODE_EMBEDDING_SIZE)
        self.fc2 = nn.Linear(CODE_EMBEDDING_SIZE, mlp_output)

    def forward(self, x):
        x = self.embed(x)
        #make CBOW ("bag of codes") representation
        x = x.sum(dim=1)
        x = x.transpose(1, 2)
        x = x.squeeze(dim=2)

        #dense
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
#        return F.relu(x)

class VanillaConv(nn.Module):

    def __init__(self, Y, kernel_size, num_filter_maps):
        super(VanillaConv, self).__init__()
        #embed
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        #load pretrained embeddings
        W = torch.Tensor(extract_wvs.load_embeddings(Y, "processed"))

        #normalize them
        sizes = W.mul(W).sum(dim=1)
        #more pytorch-y way to do this?
        for i in range(W.size()[0]):
            W[i] = W[i].mul(1./math.sqrt(sizes[i][0]))

        self.embed.weight.data.copy_(W)
        print(self.embed.weight.data)

        #xavier_uniform(self.embed.weight)
        #print(self.embed.weight.data)

        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)
        
        #conv
        self.conv = nn.Conv1d(EMBEDDING_SIZE, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)
        self.conv_drop = nn.Dropout(p=DROPOUT_DENSE)

        #dense
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/pool
#        x = self.conv_drop(self.conv(x))
        p = False
        if random.random() > 0.9999999999999:
            p = True
            print(x)
            print(x.size())
        x = self.conv(x)
        if p:
            print(x)
            print(x.size())
        x = F.relu(F.max_pool1d(x, kernel_size=x.size()[2]))
        if p:
            print(x)
            print(x.size())
#        import sys
#        sys.exit(0)
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
