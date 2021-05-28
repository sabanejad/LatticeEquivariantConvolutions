#!/usr/bin/python3


# function to grab variables from torch
def grab(var):
    return var.detach().cpu().numpy()

# The action of a phi^4 theory. Depends on parameters M2 (mass squared)
# and lam (coupling constant lambda). When called with a set of configurations
# as input, it returns the log of the probability of these configurations
# occuring (the true probability), which is used in KL loss
class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam
    def __call__(self, cfgs):
        # potential term
        action_density = self.M2*cfgs**2 + self.lam*cfgs**4 
        # kinetic term (discrete Laplacian)
        Nd = len(cfgs.shape)-1
        dims = range(1,Nd+1)
        for mu in dims:
            action_density += 2*cfgs**2
            action_density -= cfgs*torch.roll(cfgs, -1, mu)
            action_density -= cfgs*torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim=tuple(dims))

# Probability class for our prior.
# The prior is defined before training with loc = 0 and var = 1
# The log_prob returns the log of the prior (or flowed prior) probability
# for a set of configurations to occur, which is used in KL loss
# sample_n draws a sample of size batch_size
class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)

# makes the checkerboard grid. For physical parity equal to 1,
# it's a checkerboard of 1's and 0's
def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(torch_device)


# The n_layers stacked coupling layers. The make_conv_net() function and AffineCoupling
# class differ between CNN, invariant CNN and ENN and are defined in the corresponding files
# The input channel is 1 for the LxL configurations and the final output channel is 2
# for the s and t functions
def make_phi4_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1, out_channels=2, hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True)
        coupling = AffineCoupling(net, mask_shape=lattice_shape, mask_parity=parity)
        layers.append(coupling)
    return torch.nn.ModuleList(layers)

# Generates a sample from the prior distribution and applies the flow to it.
# Returns the flowed configurations and the log of the flowed probability
# The forward mode for the layer differs between CNN, invariant CNN and ENN
# and is defined in AffineCoupling class of the corresponding file
def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq.to(torch_device) - logJ.to(torch_device)
    return x.to(torch_device), logq.to(torch_device)

# The KL loss function. It takes the theoretical probability and the prior (or flowed prior)
# probability and returns their difference, which is what the network is minimizing
def calc_dkl(logp, logq):
    return (logq - logp).mean()



# The following 3 functions can be utilized during or after training to build a Monte Carlo
# Markov chain based on the current state of the flowed probability and calculate its acceptance
# rate using the theoretical probability of the phi^4 action
def compute_acceptance(ensemble_size, batch_size=64):
    phi4_ens = make_mcmc_ensemble(model, phi4_action, batch_size, ensemble_size)
    return np.mean(phi4_ens['accepted'])

def serial_sample_generator(model, action, batch_size, N_samples): 
    layers, prior = model['layers'], model['prior']
    layers.eval()
    with torch.no_grad():
        x, logq, logp = None, None, None
        for i in range(N_samples):
            batch_i = i % batch_size
            if batch_i == 0:
                x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size) 
                logp = -action(x)
            yield x[batch_i], logq[batch_i], logp[batch_i]

def make_mcmc_ensemble(model, action, batch_size, N_samples):
    history = {'x' : [], 'logq' : [], 'logp' : [], 'accepted' : []}
    # build Markov chain
    sample_gen = serial_sample_generator(model, action, batch_size, N_samples)
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            accepted = True
        else:
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1).to(torch_device)
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(new_x)
        history['accepted'].append(accepted)
    return history

