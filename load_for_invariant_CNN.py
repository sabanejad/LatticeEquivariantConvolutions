#!/usr/bin/python

# Affine layers class identical to the simple CNN. See load_for_simple_CNN for description
class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = (make_checker_mask(mask_shape, mask_parity)).to(torch_device)
        self.net = net.to(torch_device)
    def forward(self, x):
        x = x.to(torch_device)
        x_frozen = (self.mask).to(torch_device) * x
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ
    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask)*(-s), dim=tuple(axes))
        return x, logJ

# Class to replace the ConvNd. The kernel is forced to have the same weights. The code
# is courtesy of Dan Hackett
class _HypercubicConvNd:
    def __init__(self, *args, force_allow_nonsquare_kernel=False, **kwargs):
        super().__init__(*args, **kwargs) # so reset_parameters hits stencil_eights
        del self.weight
        self.stencils = self.make_stencils(self.kernel_size)
        self.stencil_weights = torch.nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, self.stencils.shape[0]))
        self.reset_stencil_params()
    @staticmethod
    def make_stencils(kernel_size):
        centers = [k//2 for k in kernel_size]
        coords = np.meshgrid(*[np.arange(k) for k in kernel_size], indexing='ij')
        coords = np.stack([x-x0 for x,x0 in zip(coords,centers)], axis=0)
        dsq = (coords**2).sum(axis=0)
        stencils = np.stack([np.int_(dsq==v) for v in np.sort(np.unique(dsq))])
        return torch.tensor(stencils)
    def reset_stencil_params(self):
        torch.nn.init.kaiming_uniform_(self.stencil_weights, a=5**0.5)
    def reset_parameters(self):
        super().reset_parameters()
        # (copied from ConvNd reset_params for weight)
        if getattr(self, 'stencil_weights', None):
            self.reset_stencil_params()
    def _compute_weight(self):
        Nd = len(self.stencils.shape)-1
        s = (self.stencils.reshape((1,1) + self.stencils.shape)).to(torch_device)
        w = (self.stencil_weights.reshape(self.stencil_weights.shape + (1,)*Nd)).to(torch_device)
        self.weight = ((w*s).sum(axis=2)).to(torch_device)
    def forward(self, *args, **kwargs):
        self._compute_weight()
        return super().forward(*args, **kwargs)

# Inner convolutional layers using the new class instead of the torch built-in Conv2d
# Remember to run the following two lines before using
# class _HypercubicConv2d(torch.nn.Conv2d): pass
# class HypercubicConv2d(_HypercubicConvNd, _HypercubicConv2d): pass
def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh): 
    sizes = [in_channels] + hidden_sizes + [out_channels]
    padding_size = (kernel_size // 2)
    net = []
    for i in range(len(sizes) - 1):
        net.append(HypercubicConv2d(
            sizes[i], sizes[i+1], kernel_size, padding=padding_size, 
            stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)
