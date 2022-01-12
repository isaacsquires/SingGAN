import torch
import torch.nn as nn
import numpy as np
from .network import make_nets
from PIL import Image
import io
import base64


def evaluate(audio, ngpu=1, seed=None):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factor
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """

    _, netG = make_nets(Training=0)
    net_g = netG()

    if torch.cuda.device_count() > 1 and ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu))).to(device)
    model_state_dict = torch.load(
        'model/saved_models/Gen.pt', map_location=device)
    net_g.load_state_dict(model_state_dict)
    net_g.eval()
    if seed:
        torch.manual_seed(seed)
    audio = torch.Tensor(audio)
    # noise = torch.randn(1, nz, lf, lf, lf)
    audio = audio.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    raw = net_g(audio)
    im = Image.fromarray(
        np.uint8(raw.detach().permute(0, 2, 3, 4, 1).numpy()[0, 63])*255)
    img_byte_arr = io.BytesIO()
    im.save(img_byte_arr, 'JPEG')
    encoded_img = base64.encodebytes(
        img_byte_arr.getvalue()).decode('ascii')

    return encoded_img
