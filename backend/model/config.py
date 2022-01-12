imsize = 128
padded_size = 184
label_channels = 2
nz = 100
lz = 1
bs = 6
device_name = "cpu"
ngpu = 0
channels = 3


class Config():
    def __init__(self):
        # fake data
        self.imsize = imsize
        self.segmentation_channel = False


class TrainParams(Config):
    def __init__(self):
        super().__init__()
        self.l = imsize
        self.padded_size = padded_size
        self.nc = channels
        # Training hyperparams
        self.batch_size = bs
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.D_batch_size = 50000  # number of real lines to sample
        self.D_fake_batch_size = bs
#         self.num_epochs = 1
#         self.iters = 50
        self.num_epochs = 250
        self.iters = 1000
        self.lrg = 0.0001
        self.lr = 0.0001
        self.Lambda = 10
        self.critic_iters = 10
        self.lz = lz
        self.ngpu = ngpu
        if self.ngpu > 1:
            self.device_name = "cuda:0"
        else:
            self.device_name = device_name
        self.sf = 1
        self.pix_coeff = 10
        self.nz = nz
        self.time = False

    def get_params(self):
        return self.l, self.nc, self.batch_size, self.beta1, self.beta2, self.D_batch_size, self.D_fake_batch_size, self.num_epochs, \
            self.iters, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz, self.sf, self.pix_coeff

    def get_path(self):
        return [self.preproc_img_path]

    def get_imtype(self):
        return self.imtype


# Network Architectures


class NetParams(Config):
    def __init__(self):
        super().__init__()
        # Architecture
        self.nets_path = f'model/saved_models/'
        self.imsize, self.nz,  self.channels, self.sf = imsize, nz, channels, 1
        self.lays = 7
        self.laysd = 6
        # kernel sizes
        self.dk, self.gk = [4]*self.laysd, [4]*self.lays
        self.ds, self.gs = [2]*self.laysd, [2]*self.lays
        self.df, self.gf = [self.channels+label_channels, 4, 16, 32, 64, 128, 1], [
            self.nz, 256, 128, 64, 32, 16, 8, self.channels]
        self.dp = [1, 1, 1, 1, 1, 0]
        self.gp = [1, 1, 1, 1, 1, 1, 1]

    def get_params(self):
        return self.dk, self.ds, self.df, self.dp, self.gk, self.gs, self.gf, self.gp

    def get_path(self):
        return self.nets_path

    def get_full_path(self):
        return self.nets_path_full

    def get_imtype(self):
        return self.imtype
