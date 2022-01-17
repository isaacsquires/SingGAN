import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from .config import NetParams


def make_nets(tag='', Training=0):

    if Training:
        params = NetParams()
        dk, ds, df, dp, gk, gs, gf, gp = params.get_params()
        with open(f'model/saved_models/{tag}_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params.get_params(), filehandle)
    else:
        with open(f'model/saved_models/{tag}_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)

    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                self.convs.append(nn.ConvTranspose2d(
                    gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm2d(gf[lay+1]))

        def forward(self, x):
            for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                x = F.relu_(bn(conv(x)))

            # use tanh if colour or grayscale, otherwise softmax for one hot encoded
            out = 0.5*torch.tanh(self.convs[-1](x))+1
            return out  # bs x n x imsize x imsize x imsize

    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(
                    nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            # concat along channel dim to get (bs, nc+1, l)
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)  # bs x 1 x 1
            return x

    return DiscriminatorWGAN, GeneratorWGAN
