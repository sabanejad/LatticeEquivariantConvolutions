#!/usr/bin/python

# The ENN affine layers. The forward and backward mode are modified so that they turn
# the 1-channel LxL torch tensor input into a GeometricTensor of the e2cnn package,
# that transforms under the
# trivial representation of the C_4 group of rotations by 2pi/4
# It is run through the ENN net and the output is transformed into a regular torch tensor
class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = (make_checker_mask(mask_shape, mask_parity)).to(torch_device)
        self.net = net.to(torch_device)
    def forward(self, x):
        x = x.to(torch_device)
        x_frozen = (self.mask).to(torch_device) * x
        x_active = (1 - self.mask) * x
        x_for_ENN = x_frozen.unsqueeze(1)
        r2_act = gspaces.Rot2dOnR2(N=4)
        c_in = nn.FieldType(r2_act, [r2_act.trivial_repr])
        x_for_ENN = nn.GeometricTensor(x_for_ENN, c_in)
        net_out = (self.net(x_for_ENN)).tensor
        s, t = net_out[:,0], net_out[:,1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ
    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        fx_for_ENN = x_frozen.unsqueeze(1)
        r2_act = gspaces.Rot2dOnR2(N=4)
        c_in = nn.FieldType(r2_act, [r2_act.trivial_repr])
        fx_for_ENN = nn.GeometricTensor(fx_for_ENN, c_in)
        net_out = (self.net(fx_for_ENN)).tensor
        s, t = net_out[:,0], net_out[:,1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask)*(-s), dim=tuple(axes))
        return x, logJ

# The inner equivariant layers. The R2Conv is the e2cnn equivalent of the
# Conv2d. Everything transforms as trivial representations of the group of
# rotations by 2 pi/4. Same architecture as for CNNs except we're using
# ReLU instead of LeakyReLU since the latter is not available in this package
def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels): 
    sizes = [in_channels] + hidden_sizes + [out_channels]
    padding_size = (kernel_size // 2)
    r2_act = gspaces.Rot2dOnR2(N=4)
    net = []
    for i in range(len(sizes) - 1):
        feat_type_in  = nn.FieldType(r2_act,  sizes[i]*[r2_act.trivial_repr])
        feat_type_out = nn.FieldType(r2_act, sizes[i+1]*[r2_act.trivial_repr])
        net.append(nn.R2Conv(
            feat_type_in, feat_type_out, kernel_size, padding=padding_size, 
            stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(nn.ReLU(feat_type_out))
        else:
            net.append(nn.PointwiseNonLinearity(feat_type_out,function='p_tanh'))
    return nn.SequentialModule(*net)
