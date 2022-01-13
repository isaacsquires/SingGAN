from .network import make_nets
from .config import TrainParams, NetParams
from .util import CustomImageDataset, grad_pen
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def train():
    # Create saved model directory
    file_path = "model/saved_models"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

    print('Training')
    netD, netG = make_nets(Training=1)

    t_params = TrainParams()
    n_params = NetParams()

    l, nc, batch_size, beta1, beta2, num_epochs, \
        iters, lrg, lr, Lambda, critic_iters, lz, nz, pix_coeff = t_params.get_params()

    ngpu = t_params.ngpu
    device = torch.device(t_params.device_name if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using {ngpu} GPUs')
    print(device, " will be used.\n")

    net_g = netG().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu))).to(device)
    optG = optim.Adam(net_g.parameters(), lr=lrg, betas=(beta1, beta2))

    net_d = netD().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        net_d = (nn.DataParallel(net_d, list(range(ngpu)))).to(device)
    optD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, beta2))

    dataset = CustomImageDataset(t_params.data_dir, l)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        times = []
        for i in range(iters):

            noise = torch.randn(batch_size, nz,
                                lz, lz, device=device)
            fake_data = net_g(noise).detach()
            out_fake = net_d(fake_data).mean()
            real_data = next(iter(train_dataloader)).to(device)
            out_real = net_d(real_data).view(-1).mean()
            gradient_penalty = grad_pen(
                net_d, real_data, fake_data, l, device, Lambda, nc)

            disc_cost = out_fake - out_real + gradient_penalty
            disc_cost.backward()
            optD.step()

            if i % int(critic_iters) == 0:
                noise = torch.randn(batch_size, nz,
                                    lz, lz, device=device)
                for dim, (d1, d2, d3) in enumerate(zip([2, 3, 2], [3, 4, 4], [4, 2, 3])):
                    net_g.zero_grad(set_to_none=True)
                    # Forward pass through G with noise vector
                    fake_data = net_g(noise)
                    output = net_d(fake_data).mean()
                    # Calculate loss for G and backprop
                    G_cost = -output
                    G_cost.backward()
                    optG.step()

            if i % 50 == 0:
                print(i)
                net_g.eval()
                with torch.no_grad():
                    torch.save(net_g.state_dict(), 'model/saved_models/Gen.pt')
                    torch.save(net_d.state_dict(),
                               'model/saved_models/Disc.pt')
