"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

import math
import random
import sys
import time

sys.path.append('../')
from constants import *
from dataproc import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts[0])
            self.embed = nn.Embedding(vocab_size+2, embed_size)


    def get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            if random.random() > 0.99:
                #keep track of this while training
                print("loss: %.5f" % loss.data[0])
                print("diff: %.5f" % diff.data[0])
                print("total loss: %.5f" % (loss + diff).data[0])
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                lt = Variable(torch.cuda.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs

    def params_to_optimize(self):
        return self.parameters()

class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=kernel_size/2)
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        self.U.bias.data.fill_(0)
        self.U.bias.requires_grad = False

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        if gpu:
            self.final = Variable(torch.zeros((Y, num_filter_maps)).cuda())
            self.final_bias = Variable(torch.zeros(Y).cuda())
        else:
            self.final = Variable(torch.zeros((Y, num_filter_maps)))
            self.final_bias = Variable(torch.zeros(Y))
        xavier_uniform(self.final)
        self.final.requires_grad = True
        self.final_bias.requires_grad = True

        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1])
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=kernel_size/2)
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)
        
    def forward(self, x, target, desc_data=None, get_attention=True):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        y = []
        ss = []
        alphas = []
        for x_i in x:
            #apply attention
            alpha_i = F.softmax(self.U(x_i).t())
            #document representations are weighted sums using the attention. Can compute all at once as a matmul
            m_i = alpha_i.mm(x_i)

            #final layer classification
            y_i = self.final.mul(m_i).sum(dim=1).add(self.final_bias)

            #save attention
            alphas.append(alpha_i)
            y.append(y_i)

        r = torch.stack(y)
        alpha = torch.stack(alphas)
        
        if desc_data is not None:
            desc_data = desc_data
            #run descriptions through description module
            b_batch = self.embed_descriptions(desc_data)
            #get l2 similarity loss
            diffs = self.compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = F.sigmoid(r)
        loss = self.get_loss(yhat, target, diffs)
        return yhat, loss, alpha
    
    def params_to_optimize(self):
        #use this to include final layer parameters and exclude attention (U)'s bias term
        ps = []
        for param in self.embed.parameters():
            ps.append(param)
        for param in self.conv.parameters():
            ps.append(param)
        for i,param in enumerate(self.U.parameters()):
            #don't add bias
            if i == 0:
                ps.append(param)
        if self.lmbda > 0:
            for param in self.desc_embedding.parameters():
                ps.append(param)
            for param in self.label_conv.parameters():
                ps.append(param)
            for param in self.label_fc1.parameters():
                ps.append(param)
        ps.append(self.final)
        ps.append(self.final_bias)
        return ps

class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        loss = self.get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full


class VanillaRNN(BaseModel):
    """
        General on purpose - can be LSTM or GRU
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, self.rnn_dim/self.num_directions, self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, self.rnn_dim/self.num_directions, self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = F.sigmoid(self.final(last_hidden))
        loss = self.get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  self.rnn_dim/self.num_directions).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      self.rnn_dim/self.num_directions).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.rnn_dim/self.num_directions))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.rnn_dim/self.num_directions))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
