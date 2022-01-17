import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from .config import NetParams


def make_nets(tag='', Training=0, net_tag='dcgan_128'):

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

    if net_tag == 'wgan':
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
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

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
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

    elif net_tag == 'dcgan_128':
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.ConvTranspose2d(gf[0], 128, 4, 1, 0),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 4, 1, 'same'),
                    nn.BatchNorm2d(128, momentum=0.7),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 4, 1, 'same'),
                    nn.BatchNorm2d(64, momentum=0.7),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, 1, 'same'),
                    nn.BatchNorm2d(32, momentum=0.7),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(32, 16, 3, 1, 'same'),
                    nn.BatchNorm2d(16, momentum=0.7),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(16, 8, 3, 1, 'same'),
                    # nn.BatchNorm2d(8, momentum=0.7),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(8, 3, 3, 1, 'same'),
                    nn.Tanh(),
                )

            def forward(self, x):
                return self.model(x)

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()

                self.model = nn.Sequential(

                    GaussianNoise(0.2),
                    nn.Conv2d(3, 8, 3, 1, 'same'),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                    nn.Conv2d(8, 16, 3, 1, 'same'),
                    nn.BatchNorm2d(16, momentum=0.7),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                    nn.Conv2d(16, 32, 3, 1, 'same'),
                    nn.BatchNorm2d(32, momentum=0.7),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                    nn.Conv2d(32, 64, 3, 1, 'same'),
                    nn.BatchNorm2d(64, momentum=0.7),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 128, 3, 1, 'same'),
                    nn.BatchNorm2d(128, momentum=0.7),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                    nn.Flatten(),
                    nn.Linear(2048, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.model(x)

    return Discriminator, Generator


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(
                *x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x
