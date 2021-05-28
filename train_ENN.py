#!/usr/bin/python3

import base64
import io
import pickle
import numpy as np
import torch
import sys
torch_device = 'cuda'
float_dtype = np.float32

from load_for_invariant_CNN import *

L = int(sys.argv[1])    # lattice size as user input
blep = int(sys.argv[2]) # if 0 use KL loss optimal parameters. if 1 use acceptance optimal parameters

def train_step(model, action, loss_fn, optimizer, metrics):
    layers, prior = model['layers'], model['prior']
    layers.train()
    optimizer.zero_grad()
    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
    loss.backward()
    optimizer.step()
    metrics['loss'].append(grab(loss))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))

# grab the corresponding best parameters from the saved file
HyperResults = np.load("data/L"+str(L)+"_hyper_ENN.npy")
nozeros = np.ma.masked_equal(HyperResults, 0.0, copy=False)
if blep == 0:
    best = np.where((nozeros[:,:,:,:,0]==np.min(nozeros[:,:,:,:,0])))
    mystring = "best_loss"
if blep == 1:
    best = np.where((nozeros[:,:,:,:,1]==np.max(nozeros[:,:,:,:,1])))
    mystring = "best_acceptance"

# ordered parameters as used in the hyperparameter files.
# order is layers, hidder_size, hidden_number, batch_size
paramlist = [[8,16,24,32],[4,8,12,16],[1,2,3,4],[64,128,256,512]]

# set up all parameters
lattice_shape = (L,L)
M2 = -4.0
lam = 8.0
phi4_action = ScalarPhi4Action(M2=M2, lam=lam)
prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
n_layers = paramlist[0][best[0][0]]
hidden_sizes = [paramlist[1][best[1][0]]]*paramlist[2][best[2][0]]
batch_size = paramlist[3][best[3][0]]
N_era = 600   # note 4 times larger than 150 used for CNNs since rate takes longer to plateau
N_epoch = 100
kernel_size = 3
layers = make_phi4_affine_layers(
    lattice_shape=lattice_shape, n_layers=n_layers,
    hidden_sizes=hidden_sizes, kernel_size=kernel_size)
model = {'layers': layers, 'prior': prior}
base_lr = 0.0038
optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)

# initialize history
history = { 'loss' : [],
        'logp' : [], 'logq' : []
        }
# train, saving loss every era. Additionally use the trained model to compute acceptance
# for every era
best_metrics = np.zeros((N_era,2)) # array to save loss and acceptance
for era in range(N_era):
    for epoch in range(N_epoch):
        train_step(model, phi4_action, calc_dkl, optimizer, history)
        if epoch % N_epoch == 0:
            best_metrics[era,0] = history['loss'][-1]
            best_metrics[era,1] = compute_acceptance(8192)

# save metrics and model
np.save("data/L"+str(L)+"_"+mystring+"_ENN.npy",best_metrics)
torch.save(model['layers'].state_dict(), "data/L"+str(L)+"_"+mystring+"_ENN.pt")
