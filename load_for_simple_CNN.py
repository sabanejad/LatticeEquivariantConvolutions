# The outer coupling layer class (note this could be multiple such layers stacked).
# The forward mode takes single-channel input x,
# the LxL field configurations, and uses half of them in a checkerboard fashion
# to run them through the inner neural net and produce a 2-channel output for s and t
# It then returns the flowed field configurations and the log of their probability
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

# The inner CNN layers. The number of hidden layers and their channel size are tunable
# parameters. Each convolutional layer is followed by a ReLU that is allowed to take
# slightly negative values, except for the last non-linear layer which is a tanh
# For the simple CNN case we use the torch built-in Conv2d
def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh): 
    sizes = [in_channels] + hidden_sizes + [out_channels]
    padding_size = (kernel_size // 2)
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Conv2d(
            sizes[i], sizes[i+1], kernel_size, padding=padding_size, 
            stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)
