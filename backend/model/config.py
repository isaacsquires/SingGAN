imsize = 128
nz = 256
lz = 1
bs = 64
device_name = "cuda:1"
ngpu = 1
channels = 3
data_dir = 'model/data/Hands'


class Config():
    def __init__(self):
        # fake data
        self.imsize = imsize
        self.segmentation_channel = False
        self.data_dir = data_dir


class TrainParams(Config):
    def __init__(self):
        super().__init__()
        self.l = imsize
        self.nc = channels
        # Training hyperparams
        self.batch_size = bs
        self.beta1 = 0.9
        self.beta2 = 0.999
#         self.num_epochs = 1
#         self.iters = 50
        self.num_epochs = 10000
        self.iters = 100
        self.lrg = 0.0001
        self.lr = 0.0001
        self.Lambda = 10
        self.critic_iters = 1
        self.lz = lz
        self.ngpu = ngpu
        if self.ngpu > 1:
            self.device_name = "cuda:0"
        else:
            self.device_name = device_name
        self.pix_coeff = 10
        self.nz = nz
        self.time = False

    def get_params(self):
        return self.l, self.nc, self.batch_size, self.beta1, self.beta2, self.num_epochs, \
            self.iters, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz, self.pix_coeff

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
        self.df, self.gf = [self.channels, 4, 16, 32, 64, 128, 1], [
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
