"""
    Test wild Theano shit
"""

from keras import backend as K
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as  np
import random

def warp_loss(y_true, y_hat):
    """
        WARP loss from WSABIE paper.
        what is the margin?
        params: 
            y_true: binary vector of correct labels
            y_hat: probability vector
        returns:
            loss: the loss
    """
    #TODO: make it Theano-differentiable
    #TODO: test
    #TODO: do we want to do a simple average? doing this at first naively
    #TODO: margin? y_hat is in [0, 1]
    margin = 1
    true_labels = np.nonzero(y_true)
        #if we were multiclass, we'd just return L(Y-1/N)*hinge(y, ybar)
        #instead add it to an array
#        losses.append((1/N)*K.maximum(margin - f + f_bar, 0.))
    N = np.int(0)
    margin = np.int(1)
    Ns, _ = theano.scan(
            fn=sample,
            n_steps=len(y_true)-1,
            outputs_info=[true_labels],
            non_sequences=[y_hat, N, margin]
    )
    losses, _ = theano.scan(
            fn=w
    )
    return K.mean(losses)

def sample(true_label, N, prev_margin, y_hat):
    #pick random labels until you find one with prob > correct label - margin
    compl = [i for i in range(10) if i != true_label]
    rng = RandomStreams(seed=123)
    label = rng.choice(compl)
    label = 5
    f_bar = y_hat[label]
    print(type(true_label))
    f = y_hat[true_label]
    N += 1
    condition = T.lt(f - margin, f_bar)
    return N, margin - f + f_bar, theano.scan_module.until(condition)

#variables
y_true = T.ivector('y_true')
y_hat = T.fvector('y_hat')
true_labels = T.fvector("true_labels")
#compls = T.imatrix("compls")
#N = T.ivector("N")
N = T.iscalar("N")
margin = T.fscalar('margin')

#     losses.append((1/N)*K.maximum(margin - f + f_bar, 0.))
#sample scan. scan over true labels vector. take in y_hat, margin as non-sequences b/c they don't change
#N, margin are the expected outputs
outputs, updates = theano.scan(
        fn=sample,
        n_steps = 9,
        outputs_info=[np.float64(0), np.float64(0)],
        sequences=[true_labels],
        non_sequences=[y_hat],
)

loss = theano.function(
        inputs=[y_hat, N, margin, true_labels],
        outputs=outputs,
)
margin_inp = np.float64(1.)
#N_inp = np.array([range(10)])
N_inp = np.int(0)
y_true_inp = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
y_hat_inp = np.array([.3, .7, .6, .4, .2, .8, .2, .3, .4, .6])
true_labels_inp = np.nonzero(y_true_inp)
print("true labels: " + str(true_labels_inp))
#compl = np.array([[i for i in range(10) if i != true_label] for true_label in true_labels_inp])
#print("compl: " + str(compl))
example_loss = loss(y_hat_inp, N_inp, margin_inp, true_labels_inp)
