"""
    class for warp loss
"""
import torch
from torch.autograd import Function

import math
import random

class WarpLoss(Function):

    def __init__(self, output, target, margin=1, gpu=True):
        super(WarpLoss, self).__init__()
        self.margin = margin
        self.gpu = gpu
        #precompute L
        #copy output to output_buf variable for sample/precompute stuff
        output_buf = output.clone()
        self.L, self.ybar, self.lowest_true, self.lowest_true_ind = self.precompute_L(output_buf, target)

    def forward(self, output):
        """
            Calculate warp loss (Weston, Bengio, Usunier 2011) L*(1-f_y+f_v)_+
            Arguments: 
                output: Variable predicted output
            Returns:
                loss: a Variable that we can call backward() on to backprop thru the graph
        """
        if self.ybar is not None:
            if self.gpu:
                ybar_t = torch.cuda.FloatTensor(output.size()[0]).zero_()
            else:
                ybar_t = torch.FloatTensor(output.size()[0]).zero_()
            ybar_t[self.ybar] = 1
            #hinge
            loss = ((output + self.margin - self.lowest_true) * ybar_t).clamp(min=0)
            loss = (loss * self.L)
            self.save_for_backward(output)
        else:
            #return a 1x1 tensor. when training, check size of returned loss and if it's 1x1, don't backprop
            if self.gpu:
                loss = torch.cuda.LongTensor([0])
            else:
                loss = torch.LongTensor([0])
        return loss

    def backward(self, grad_output):
        #gradient is self.L * -1 for index of lowest_true, self.L * 1 for index of ybar
        #ignore grad_output since this is the leaf of the graph
        output = self.saved_tensors[0]
        #initialize gradients w.r.t. output
        if self.gpu:
            grad = torch.cuda.FloatTensor(output.size()[0]).fill_(self.L)
        else:
            grad = torch.FloatTensor(output.size()[0]).fill_(self.L)

        if self.gpu:
            mask = torch.cuda.FloatTensor(output.size()[0]).zero_()
        else:
            mask = torch.FloatTensor(output.size()[0]).zero_()
        mask[self.ybar] = 1
        mask[self.lowest_true_ind] = -1

        return grad * mask

    def precompute_L(self, output_buf, target):
        buf_data = output_buf.data
        target_data = target.data
        nz = target_data == 1
        if self.gpu:
            lowest_true = buf_data.index(target_data.type(torch.cuda.ByteTensor)).min()
        else:
            lowest_true = buf_data.index(target_data.type(torch.ByteTensor)).min()
        for i,d in enumerate(buf_data):
            if d == lowest_true and nz[i]:
                lowest_true_ind = i
        z = target_data == 0
        N, ybar = self.sample(target, z, buf_data, lowest_true)
        k = int(math.floor((target.size()[0] - target_data.sum()) / N))
        return self.L_func(k), ybar, lowest_true, lowest_true_ind

    def sample(self, target, z, buf_data, lowest_true):
        #sample non-target values until we find one we predicted with more confidence than least confident value
        N = 0
        while N < (target.size()[0] - target.sum()).data[0]:
            ybar = random.choice(range(len(buf_data)))
            if z[ybar]:
                if buf_data[ybar] >= lowest_true - self.margin:
                    return N+1, ybar
                N += 1
        return N, None

    def L_func(self, k):
        return sum([1./(i+1) for i in range(k)])

def warp_loss(output, target):
    return WarpLoss(output, target)(output)
