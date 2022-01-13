from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os
from torch import autograd


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, imsize=128, transform=None):
        self.img_dir = img_dir
        self.imsize = imsize
        self.img_labels = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        try:
            image = read_image(img_path)
        except:
            print(img_path)
        if self.imsize:
            image = interpolate(image.unsqueeze(
                0), (self.imsize, self.imsize))[0]/255
        return image


def grad_pen(netD, real_data, fake_data, l, device, gp_lambda, nc):
    """
    Possible approaches:
    1. Interpolate labels (doesnt work)
    2. Find cases where labels match (doesnt work, no matching labels)
    3. Slice fake data through center to make 2D and take lines from there
    4. Round values of labels
    """
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    # alpha_labels = alpha.clone()[:, 0].unsqueeze(1)
    alpha = alpha.to(device)
    # alpha_labels = alpha_labels.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty
