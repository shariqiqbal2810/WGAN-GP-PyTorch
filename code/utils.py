import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import grad, Variable
from scipy.io import loadmat
from os.path import expanduser


class SVHNDataset(Dataset):
    """
    Street View House Numbers PyTorch Dataset.
    Takes .mat file provided at http://ufldl.stanford.edu/housenumbers/
    and prepares for ingestion into PyTorch model
    """
    def __init__(self, data_loc):
        """
        Inputs:
            data_loc (string): Location of SVHN .mat data file
        """
        data_dict = loadmat(expanduser(data_loc))
        X_data_raw = np.transpose(data_dict['X'], (3, 2, 1, 0))
        X_data = (X_data_raw.astype("float32") / 255) * 2 - 1
        self.X_data = X_data
        self.num_pixels = np.prod(self.X_data.shape[1:])
        self.image_shape = X_data.shape[1:]

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx]


class Discriminator(nn.Module):
    """
    General Discriminator for small dataset image GAN models
    """
    def __init__(self, image_shape=(3, 32, 32), dim_factor=64):
        """
        Inputs:
            image_shape (tuple of int): Shape of input images (H, W, C)
            dim_factor (int): Base factor to use for number of hidden
                              dimensions at each layer
        """
        super(Discriminator, self).__init__()
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        H_out = int(H / 2**3)  # divide by 2^3 bc 3 convs with stride 2
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        W_out = int(W / 2**3)  # divide by 2^3 bc 3 convs with stride 2

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(C, dim_factor, 5, stride=2)
        self.conv2 = nn.Conv2d(dim_factor, 2 * dim_factor, 5,
                               stride=2)
        self.conv3 = nn.Conv2d(2 * dim_factor, 4 * dim_factor, 5,
                               stride=2)
        self.linear = nn.Linear(4 * dim_factor * H_out * W_out, 1)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Tensor): Batch of images to predict real or fake on
        Outputs:
            disc_out (PyTorch Vector): Vector of classification values for each
                                       input image (higher for more real, lower
                                       for more fake)
        """
        H1 = F.leaky_relu(self.conv1(self.pad(X)), negative_slope=0.2)
        H2 = F.leaky_relu(self.conv2(self.pad(H1)), negative_slope=0.2)
        H3 = F.leaky_relu(self.conv3(self.pad(H2)), negative_slope=0.2)
        H3_resh = H3.view(H3.size(0), -1)  # reshape for linear layer
        disc_out = self.linear(H3_resh)
        return disc_out


class Generator(nn.Module):
    """
    General Generator for small dataset image GAN models
    """
    def __init__(self, image_shape=(3, 32, 32), noise_dim=128, dim_factor=64):
        """
        Inputs:
            image_shape (tuple of int): Shape of output images (H, W, C)
            noise_dim (int): Number of dimensions for input noise
            dim_factor (int): Base factor to use for number of hidden
                              dimensions at each layer
        """
        super(Generator, self).__init__()
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        self.H_init = int(H / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        self.W_init = int(W / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5

        self.linear = nn.Linear(noise_dim,
                                4 * dim_factor * self.H_init * self.W_init)
        self.deconv1 = nn.ConvTranspose2d(4 * dim_factor, 2 * dim_factor,
                                          4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * dim_factor, dim_factor,
                                          4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(dim_factor, C,
                                          4, stride=2, padding=1)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Tensor): Random noise input
        Outputs:
            img_out (PyTorch Tensor): Generated batch of images
        """
        H1 = F.relu(self.linear(X))
        H1_resh = H1.view(H1.size(0), -1, self.W_init, self.H_init)
        H2 = F.relu(self.deconv1(H1_resh))
        H3 = F.relu(self.deconv2(H2))
        img_out = F.tanh(self.deconv3(H3))
        return img_out


def wgan_generator_loss(gen_noise, gen_net, disc_net):
    """
    Generator loss for Wasserstein GAN (same for WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
    Outputs:
        loss (PyTorch scalar): Generator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out = disc_net(gen_data)
    # get loss
    loss = -disc_out.mean()
    return loss


def wgan_gp_discriminator_loss(gen_noise, real_data, gen_net, disc_net, lmbda,
                               gp_alpha):
    """
    Discriminator loss with gradient penalty for Wasserstein GAN (WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        real_data (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
        lmbda (float): Hyperparameter for weighting gradient penalty
        gp_alpha (PyTorch Tensor): Values to use to randomly interpolate
                                   between real and fake data for GP penalty
    Outputs:
        loss (PyTorch scalar): Discriminator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out_gen = disc_net(gen_data)
    disc_out_real = disc_net(real_data)
    # get loss (w/o GP)
    loss = disc_out_gen.mean() - disc_out_real.mean()
    # draw interpolation values
    gp_alpha.uniform_()
    # interpolate between real and generated data
    interpolates = gp_alpha * real_data.data + (1 - gp_alpha) * gen_data.data
    interpolates = Variable(interpolates, requires_grad=True)
    # feed interpolates through discriminator
    disc_out_interp = disc_net(interpolates)
    # get gradients of discriminator output with respect to input
    gradients = grad(outputs=disc_out_interp.sum(), inputs=interpolates,
                     create_graph=True)[0]
    # calculate gradient penalty
    grad_pen = ((gradients.view(gradients.size(0), -1).norm(
        2, dim=1) - 1)**2).mean()
    # add gradient penalty to loss
    loss += lmbda * grad_pen
    return loss


def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False
