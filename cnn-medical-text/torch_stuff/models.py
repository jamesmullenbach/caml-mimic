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

class HAN(nn.Module):
    def __init__(self, Y, word_lstm_dim, doc_lstm_dim, batch_size, gpu):
        super(HAN, self).__init__() 
        self.word_attn = WordLSTMAttn(Y, word_lstm_dim, gpu, 1)
        self.doc_lstm_dim = doc_lstm_dim
        self.batch_size = batch_size
        self.gpu = gpu

        self.doc_lstm = nn.LSTM(word_lstm_dim, doc_lstm_dim, num_layers=1, bidirectional=False)
        self.doc_lin = nn.Linear(doc_lstm_dim, doc_lstm_dim)
        self.query_vec = nn.Linear(doc_lstm_dim, 1)
        self.query_vec.bias.data.fill_(0)

        self.final_layer = nn.Linear(doc_lstm_dim, Y)

        self.doc_hidden = self.init_hidden()

    def init_hidden(self): 
        if self.gpu:
            return (Variable(torch.cuda.FloatTensor(1, self.batch_size, self.doc_lstm_dim).zero_()),
                           Variable(torch.cuda.FloatTensor(1, self.batch_size, self.doc_lstm_dim).zero_()))
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.doc_lstm_dim)),
                           Variable(torch.zeros(1, self.batch_size, self.doc_lstm_dim)))       

    def refresh(self, batch_size):
        self.word_attn.refresh(1)
        self.batch_size = batch_size
        self.doc_hidden = self.init_hidden()

    def forward(self, x, get_importances=False, volatile=False):
        """
            Input: list of note lists (same # notes in each list for batching)
        """
        docs_data = []
        word_sims = []
        for i in range(x.shape[0]):
            #send the notes through word LSTM to get s_i's (note vectors), cat them together into a sequence
            if not get_importances:
                if self.gpu:
                    docs_data.append(torch.stack([self.word_attn(Variable(torch.cuda.LongTensor(x_i), volatile=volatile).unsqueeze(0)) \
                                                for x_i in x[i]], 0).unsqueeze(1))
                else:
                    docs_data.append(torch.stack([self.word_attn(Variable(torch.LongTensor(x_i), volatile=volatile).unsqueeze(0)) \
                                                for x_i in x[i]], 0).unsqueeze(1))
            else:
                if self.gpu:
                    word_attn_res = [self.word_attn(Variable(torch.cuda.LongTensor(x_i), volatile=volatile).unsqueeze(0), get_importances)
                                     for x_i in x[i]]
                    docs_data.append(torch.stack([res[0] for res in word_attn_res]).unsqueeze(1))
                    word_sims.append([res[1].squeeze() for res in word_attn_res])
                else:
                    word_attn_res = [self.word_attn(Variable(torch.LongTensor(x_i), volatile=volatile).unsqueeze(0), get_importances)
                                     for x_i in x[i]]
                    docs_data.append(torch.stack([res[0] for res in word_attn_res]).unsqueeze(1))
                    word_sims.append([res[1].squeeze() for res in word_attn_res])

        #at this point, word_sims is [batch_size x num_notes] array. each entry is note_length vector of word importances
        #put all the note vector sequences into one batch. result is num_notes x batch_size x word_lstm_dim
        note_vecs = torch.cat(docs_data, 1)
        num_notes, batch_size, _ = note_vecs.size()
        h, self.doc_hidden = self.doc_lstm(note_vecs, self.doc_hidden)
       
        #linear layer (for all hidden layers of lstm)
        u_i = F.tanh(self.doc_lin(h.view(-1, self.doc_lstm_dim)))
        #dot product similarity
        doc_sims = self.query_vec(u_i).view(-1, num_notes)
        #softmax to get attn vector
        attn = F.softmax(doc_sims)
        #weighted sum. resulting size is batch_size x doc_lstm_dim
        h = h.t()
        attn = attn.unsqueeze(1)
        wsum = torch.cat([attn[i].mm(h[i]) for i in range(batch_size)])
        if get_importances:
            return self.final_layer(wsum), word_sims, doc_sims
        else:
            return self.final_layer(wsum)

class WordLSTMAttn(nn.Module):
    def __init__(self, Y, word_lstm_dim, gpu, batch_size):
        super(WordLSTMAttn, self).__init__() 
        self.word_lstm_dim = word_lstm_dim
        self.batch_size = batch_size
        self.gpu = gpu
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
        self.embed.weight.data = W.clone()
       
        self.word_lstm = nn.LSTM(W.size()[1], word_lstm_dim/2, 1, bidirectional=True)
        self.word_lin = nn.Linear(word_lstm_dim, word_lstm_dim)
        self.query_vec = nn.Linear(word_lstm_dim, 1)
        self.query_vec.bias.data.fill_(0)

        self.word_hidden = self.init_hidden()
 
    def init_hidden(self):
        if self.gpu:
            return (Variable(torch.cuda.FloatTensor(2, self.batch_size, self.word_lstm_dim/2).zero_()),
                           Variable(torch.cuda.FloatTensor(2, self.batch_size, self.word_lstm_dim/2).zero_()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.word_lstm_dim/2)),
                           Variable(torch.zeros(2, self.batch_size, self.word_lstm_dim/2)))   

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.word_hidden = self.init_hidden()

    def forward(self, x, get_importances=False):
        """
            Input: batch together inputs by number of notes?
            pass each note through word-attention biLSTM, then doc-attention uniLSTM
        """
        #x_it = embedded word t
        #h_it = bidirectional GRU/LSTM dimensions
        #u_it = tanh(W*h_it + b) <- linear on hidden layers in recurrent layer
        #alpha_it = softmax(u_it's dot u_w) <- dot each linear output with query vec
        #s_i = sum (alpha_it * h_it)
        embeds = self.embed(x).view(x.size()[1], 1, -1)
        h, self.word_hidden = self.word_lstm(embeds, self.word_hidden)
        #because batch is 1, I can squeeze the batch dim and compute next two parts over the words as a batch
        h = h.squeeze(1)
       
        #linear layer
        u_i = F.tanh(self.word_lin(h))
        #dot product similarity.
        sims = self.query_vec(u_i).squeeze()
        #softmax to get attn vector
        attn = F.softmax(sims)
        #return attn-weighted sum with lstm "word annotations"
        wsum = h.t().mv(attn)
        if get_importances:
            return wsum, sims
        else:
            return wsum

class VanillaLSTM(nn.Module):
    def __init__(self, Y, lstm_dim, gpu):
        super(VanillaLSTM, self).__init__()
        self.gpu = gpu
        self.lstm_dim = lstm_dim
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
        self.lstm = nn.LSTM(W.size()[1], self.lstm_dim/2, 1, bidirectional=True)
        self.final_layer = nn.Linear(self.lstm_dim, Y)

        self.batch_size = 1
        self.hidden = self.init_hidden()

    def forward(self, x, start_inds = None):
        embeds = self.embed(x).t()
        out, self.hidden = self.lstm(embeds, self.hidden)
        return self.final_layer(self.hidden[1].view(self.batch_size,-1))

    def init_hidden(self):
        if self.gpu:
            return (Variable(torch.cuda.FloatTensor(2, self.batch_size, self.lstm_dim/2).zero_()),
                           Variable(torch.cuda.FloatTensor(2, self.batch_size, self.lstm_dim/2).zero_()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.lstm_dim/2)),
                           Variable(torch.zeros(2, self.batch_size, self.lstm_dim/2)))

    def enforce_norm_constraint(self):
        pass

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

class MultiConv(nn.Module):
    
    def __init__(self, Y, min_filter, max_filter, num_filter_maps, s):
        super(MultiConv, self).__init__()
        self.s = s
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
        self.conv_drop = nn.Dropout(p=DROPOUT_DENSE)

        self.fc = nn.Linear(num_filter_maps*(max_filter-min_filter+1), Y)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)
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
                pool_outputs.append(pool_output.squeeze(dim=2))
        x = torch.cat(pool_outputs, 1)

        #dense
        x = self.fc(self.fc_drop(x))
        return x

    def enforce_norm_constraint(self):
        self.fc.weight.data.renorm(p=2, dim=0, maxnorm=self.s)
        for conv in self.convs:
            conv.weight.data.renorm(p=2, dim=0, maxnorm=self.s)

class LogReg(nn.Module):

    def __init__(self, Y):
        super(LogReg, self).__init__()
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

        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)

        self.fc = nn.Linear(EMBEDDING_SIZE, Y)

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        #make CBOW representation
        x = x.sum(dim=1)
        x = x.transpose(1, 2)
        x = x.squeeze(dim=2)

        #dense
        x = self.fc(x)

        return x

class MLP(nn.Module):

    def __init__(self, Y):
        super(MLP, self).__init__()
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

        self.embed_drop = nn.Dropout(p=DROPOUT_EMBED)

        self.fc1 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.fc2 = nn.Linear(EMBEDDING_SIZE, Y)

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        #make CBOW representation
        x = x.sum(dim=1)
        x = x.transpose(1, 2)
        x = x.squeeze(dim=2)

        #dense
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return x

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
