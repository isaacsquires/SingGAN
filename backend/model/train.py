from .network import make_nets
import torch
import pathlib


def train():
    # Create saved model directory
    file_path = "model/saved_models"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

    print('Training')
    netD, netG = make_nets(Training=1)

    net_g = netG()
    net_d = netD()

    net_g.eval()
    with torch.no_grad():
        torch.save(net_g.state_dict(), 'model/saved_models/Gen.pt')
        torch.save(net_d.state_dict(), 'model/saved_models/Disc.pt')
