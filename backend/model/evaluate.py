import torch
import torch.nn as nn
import numpy as np
from .network import make_nets
from PIL import Image
import io
import base64
import time


def evaluate(audio, ngpu=1, tag=''):
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
    tic = time.time()
    _, netG = make_nets(Training=0, tag=tag)
    net_g = netG()

    if torch.cuda.device_count() > 1 and ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu))).to(device)
    model_state_dict = torch.load(
        f'model/saved_models/{tag}_Gen.pt', map_location=device)
    net_g.load_state_dict(model_state_dict, strict=False)
    net_g.eval()
    with torch.no_grad():
        audio = torch.Tensor(audio)
        # audio = torch.randn(1, 256, 1, 1)
        audio = audio*50
        audio = audio.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        im = net_g(audio)
        im = np.uint8(im.detach().permute(0, 2, 3, 1).cpu().numpy()[0]*255)
        im = Image.fromarray(im)

        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, 'JPEG')
        encoded_img = base64.encodebytes(
            img_byte_arr.getvalue()).decode('ascii')
        toc = time.time()

        # print(f'Time: {toc-tic}')

    return encoded_img


def test_img(ngpu=2, tag=''):

    _, netG = make_nets(tag=tag, Training=0)
    net_g = netG()

    if torch.cuda.device_count() > 1 and ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu))).to(device)
    model_state_dict = torch.load(
        f'model/saved_models/{tag}_Gen.pt', map_location=device)
    net_g.load_state_dict(model_state_dict, strict=False)
    net_g.eval()
    with torch.no_grad():

        noise = torch.randn(1, 256, 1, 1)
        im2 = net_g(noise)
        im2 = np.uint8(im2.detach().permute(0, 2, 3, 1).cpu().numpy()[0]*255)
        im2 = Image.fromarray(im2)
        im2.save('example.jpeg', 'JPEG')
