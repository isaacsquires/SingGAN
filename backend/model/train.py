from .network import make_nets
from .config import TrainParams, NetParams
from .util import CustomImageDataset, grad_pen
import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
import pathlib
from torch.utils.data import DataLoader
import numpy as np
import wandb
import subprocess
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

mode = 'disabled'
# mode = None


def wandb_init(name):
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    print('initing')
    wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        # 'watch_called': False,
        'no_cuda': False,
        # 'seed': 42,
        'log_interval': 1000,

    }
    # wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    # wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']


def train(tag=''):
    wandb_init(tag)
    # Create saved model directory
    file_path = "model/saved_models"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

    print('Training')
    netD, netG = make_nets(Training=1, tag=tag)

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

    wandb.watch(net_g)
    wandb.watch(net_d)

    summary(net_d)
    summary(net_g)
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    for epoch in range(num_epochs):
        times = []
        for i in range(iters):

            net_d.zero_grad()

            real_data = next(iter(train_dataloader)).to(device)
            # noise = torch.randn(batch_size, nz,
            # lz, lz, device=device)
            # fake_data = net_g(noise).detach()
            # out_real = net_d(real_data).view(-1).mean()
            # out_fake = net_d(fake_data).mean()

            output = net_d(real_data).view(-1)
            label = torch.full((batch_size,), real_label,
                               dtype=torch.float, device=device)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz,
                                lz, lz, device=device)
            fake_data = net_g(noise).detach()
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = net_d(fake_data.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optD.step()

            wandb.log({"D(real)": D_x})
            wandb.log({"D(fake)": D_G_z1})
            wandb.log({"errD": errD})

            # gradient_penalty = grad_pen(
            #     net_d, real_data, fake_data, l, device, Lambda, nc)
            # disc_cost = out_fake - out_real + gradient_penalty

            # disc_cost.backward()

            # optD.step()

            # wandb.log({"D(real)": out_real.item()})
            # wandb.log({"D(fake)": out_fake.item()})
            # wandb.log({"Wass": (out_real-out_fake).item()})

            if i % int(critic_iters) == 0:
                net_g.zero_grad()
                noise = torch.randn(batch_size, nz,
                                    lz, lz, device=device)
                # Forward pass through G with noise vector
                fake_data = net_g(noise)
                label.fill_(real_label)
                output = net_d(fake_data).view(-1)
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optG.step()

                # Calculate loss for G and backprop
                # fake_data = net_g(noise)
                # output = net_d(fake_data).mean()
                # G_cost = -output
                # G_cost.backward()
                # optG.step()

            if i % 50 == 0:
                with torch.no_grad():
                    print(
                        f'iteration {i} of {iters}, epoch {epoch} of {num_epochs}')
                    noise = torch.randn(batch_size, nz,
                                        lz, lz, device=device)
                    out_fake = net_g(noise)[0].permute(
                        1, 2, 0).detach().cpu().numpy()
                    fig = plt.figure()
                    plt.subplot(211)
                    plt.imshow(np.uint8(out_fake*255))
                    plt.subplot(212)
                    plt.imshow(np.uint8((real_data[0]*255).permute(
                        1, 2, 0).detach().cpu().numpy()))
                    # im.save('example_fake.jpeg')
                    # im2.save('example_real.jpeg')
                    wandb.log({"Real vs fake": wandb.Image(fig)})
                    plt.close()

                    torch.save(net_g.state_dict(),
                               f'model/saved_models/{tag}_Gen.pt')
                    torch.save(net_d.state_dict(),
                               f'model/saved_models/{tag}_Disc.pt')
                    wandb.save(f'model/saved_models/{tag}_Disc.pt')
                    wandb.save(f'model/saved_models/{tag}_Gen.pt')
