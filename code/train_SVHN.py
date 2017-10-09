import torch
import argparse
import torch.optim as optim
import numpy as np
from os import makedirs
from os.path import join, exists
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import *

use_cuda = torch.cuda.is_available()

# seed for replicability
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed(SEED)


def train_model(data_loc, model_name, noise_dim=128, dim_factor=64,
                K=5, early_K=100, lmbda=10., batch_size=64, n_epochs=140,
                learning_rate=1e-4):
    # create folder to store model results
    model_folder = join('models', model_name)
    if not exists(model_folder):
        makedirs(model_folder)
    # load dataset
    dataset = SVHNDataset(data_loc)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           drop_last=True, num_workers=3,
                           pin_memory=True)
    # create networks
    gen_net = Generator(image_shape=dataset.image_shape, noise_dim=noise_dim,
                        dim_factor=dim_factor)
    disc_net = Discriminator(image_shape=dataset.image_shape,
                             dim_factor=dim_factor)
    # initialize optimizers
    gen_optimizer = optim.Adam(gen_net.parameters(), lr=learning_rate,
                               betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=learning_rate,
                                betas=(0.5, 0.9))
    # create tensors for input to algorithm
    gen_noise_tensor = torch.FloatTensor(batch_size, noise_dim)
    gp_alpha_tensor = torch.FloatTensor(batch_size, 1, 1, 1)
    # convert tensors and parameters to cuda
    if use_cuda:
        gen_net = gen_net.cuda()
        disc_net = disc_net.cuda()
        gen_noise_tensor = gen_noise_tensor.cuda()
        gp_alpha_tensor = gp_alpha_tensor.cuda()
    # wrap noise as variable so we can backprop through the graph
    gen_noise_var = Variable(gen_noise_tensor, requires_grad=False)
    # calculate batches per epoch
    bpe = len(dataset) // batch_size
    # create lists to store training loss
    gen_loss = []
    disc_loss = []
    # iterate over epochs
    for ie in range(n_epochs):
        print("-> Entering epoch %i out of %i" % (ie + 1, n_epochs))
        # iterate over data
        for ib, X_data in enumerate(data_iter):
            # wrap data in torch Tensor
            X_tensor = torch.Tensor(X_data)
            if use_cuda:
                X_tensor = X_tensor.cuda()
            X_var = Variable(X_tensor, requires_grad=False)
            # calculate total iterations
            i = bpe * ie + ib
            if ((((i % K) == (K - 1)) and i > 1000) or
                    (((i % early_K) == (early_K - 1)) and i < 1000)):
                # train generator
                enable_gradients(gen_net)  # enable gradients for gen net
                disable_gradients(disc_net)  # saves computation on backprop
                gen_net.zero_grad()
                loss = wgan_generator_loss(gen_noise_var, gen_net, disc_net)
                loss.backward()
                gen_optimizer.step()
                # append loss to list
                gen_loss.append(loss.data[0])
            # train discriminator
            enable_gradients(disc_net)  # enable gradients for disc net
            disable_gradients(gen_net)  # saves computation on backprop
            disc_net.zero_grad()
            loss = wgan_gp_discriminator_loss(gen_noise_var, X_var, gen_net,
                                              disc_net, lmbda, gp_alpha_tensor)
            loss.backward()
            disc_optimizer.step()
            # append loss to list
            disc_loss.append(loss.data[0])
        # calculate and print mean discriminator loss for past epoch
        mean_disc_loss = np.array(disc_loss[-bpe:]).mean()
        print("Mean discriminator loss over epoch: %.2f" % mean_disc_loss)
        # save model and training loss
        torch.save(gen_net.state_dict(), join(model_folder, 'gen_net.pt'))
        torch.save(disc_net.state_dict(), join(model_folder, 'disc_net.pt'))
        np.save(join(model_folder, 'gen_loss'), gen_loss)
        np.save(join(model_folder, 'disc_loss'), disc_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_loc", help="Location of SVHN data")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--noise_dim",
                        default=128, type=int,
                        help="Noise dim for generator")
    parser.add_argument("--dim_factor",
                        default=64, type=int,
                        help="Dimension factor to use for hidden layers")
    parser.add_argument("--K",
                        default=5, type=int,
                        help="Iterations of discriminator per generator")
    parser.add_argument("--lmbda",
                        default=10., type=float,
                        help="Gradient penalty hyperparameter")
    parser.add_argument("--batch_size",
                        default=64, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_epochs",
                        default=140, type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate",
                        default=1e-4, type=float,
                        help="Learning rate of the model")
    args = parser.parse_args()

    train_model(**vars(args))
