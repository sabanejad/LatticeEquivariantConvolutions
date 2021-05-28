#!/usr/bin/python3

import base64
import io
import os
import pickle
import numpy as np
import torch
import sys
import itertools
torch_device = 'cuda'
float_dtype = np.float32
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from load_for_all import *
# load all functions defined in load_for_invariant_CNN.py
from load_for_invariant_CNN import *

class _HypercubicConv2d(torch.nn.Conv2d): pass
class HypercubicConv2d(_HypercubicConvNd, _HypercubicConv2d): pass

L = int(sys.argv[1]) # lattice size provided as input when running the code

def train_step(model, action, optimizer):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()
    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
    loss.backward()
    optimizer.step()
    #return loss and acceptance computed for an ensemble of 8192 configurations
    return grab(loss), compute_acceptance(8192)

# set up common parameters
lattice_shape = (L,L)
M2 = -4.0
lam = 8.0
kernel_size = 3
base_lr = 0.003818
phi4_action = ScalarPhi4Action(M2=M2, lam=lam)
prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
N_epoch = 10

# ranges of hyperparameters to use
n_layers_list = [8,16,24,32]
hidden_size_list = [4,8,12,16]
hidden_number_list = [1,2,3,4]
batch_size_list = [64,128,256,512]

# Set up array to save loss and acceptance rate for each hyperparameter choice
HyperResults = np.zeros((len(n_layers_list),len(hidden_size_list),len(hidden_number_list)
                         ,len(batch_size_list),2))

# iterate over parameter, train for 10 epochs and save acceptance and loss
for n_layers_i,hidden_size_i,hidden_number_i,batch_size_i in itertools.product(range(len(n_layers_list)),
                                                                               range(len(hidden_size_list)),
                                                                               range(len(hidden_number_list)),
                                                                               range(len(batch_size_list))):
    n_layers = n_layers_list[n_layers_i]
    hidden_size = hidden_size_list[hidden_size_i]
    hidden_number = hidden_number_list[hidden_number_i]
    batch_size = batch_size_list[batch_size_i]
    hidden_sizes = [hidden_size]*hidden_number
    layers = make_phi4_affine_layers(lattice_shape=lattice_shape, n_layers=n_layers,
                                     hidden_sizes=hidden_sizes, kernel_size=kernel_size)
    model = {'layers': layers, 'prior': prior}
    optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)
    output = []
    for i in range(N_epoch):
        output = train_step(model,phi4_action,optimizer)
    HyperResults[n_layers_i,hidden_size_i,hidden_number_i,batch_size_i,0] = output[0]
    HyperResults[n_layers_i,hidden_size_i,hidden_number_i,batch_size_i,1] = output[1]

np.save("data/L"+str(L)+"_hyper_invariant_CNN.npy",HyperResults)
    
