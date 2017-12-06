#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:33:07 2017

@author: pc
"""
import random
from torch import optim
import os
import torchvision.datasets as dset
from torch.utils import data
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from torch.autograd import Variable
data_root = "../data/"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 64

# params for setting up models
model_root = "../model/"
num_channels = 3
z_dim = 100
d_conv_dim = 64
g_conv_dim = 64
d_model_restore = None
g_model_restore = None

# params for training
num_gpu = 1
num_epochs = 25
log_step = 10
sample_step = 100
save_step = 10
manual_seed = None


# params for optimizing models
d_steps = 1
g_steps = 1
d_learning_rate = 0.00005
g_learning_rate = 0.00005
beta1 = 0.5
beta2 = 0.999


# image pre-processing
pre_process = transforms.Compose([transforms.Scale(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=dataset_mean,
                                                       std=dataset_std)])

# dataset and data loader
dataset = dset.CIFAR10(root=data_root,
                       transform=pre_process,
                       download=True
                       )

data_loader = data.DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

class Discriminator(nn.Module):
    """Model for Discriminator."""

    def __init__(self, num_channels, conv_dim, num_gpu):
        """Init for Discriminator model."""
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.layer = nn.Sequential(
            # 1st conv layer
            # input num_channels x 64 x 64, output conv_dim x 32 x 32
            nn.Conv2d(num_channels, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd conv layer, output (conv_dim*2) x 16 x 16
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd conv layer, output (conv_dim*4) x 8 x 8
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th conv layer, output (conv_dim*8) x 4 x 4
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(conv_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Forward step for Discriminator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and \
                self.num_gpu > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_gpu))
        else:
            out = self.layer(input)
        return out.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """Model for Generator."""

    def __init__(self, num_channels, z_dim, conv_dim, num_gpu):
        """Init for Generator model."""
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.layer = nn.Sequential(
            # 1st deconv layer, input Z, output (conv_dim*8) x 4 x 4
            nn.ConvTranspose2d(z_dim, conv_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(True),
            # 2nd deconv layer, output (conv_dim*4) x 8 x 8
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(True),
            # 3rd deconv layer, output (conv_dim*2) x 16 x 16
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),
            # 4th deconv layer, output (conv_dim) x 32 x 32
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            # output layer, output (num_channels) x 64 x 64
            nn.ConvTranspose2d(conv_dim, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        """Forward step for Generator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and \
                self.num_gpu > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_gpu))
        else:
            out = self.layer(input)
        # flatten output
        return out


def get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_gpu,
               d_model_restore=None, g_model_restore=None):
    """Get models with cuda and inited weights."""
    D = Discriminator(num_channels=num_channels,
                      conv_dim=d_conv_dim,
                      num_gpu=num_gpu)
    G = Generator(num_channels=num_channels,
                  z_dim=z_dim,
                  conv_dim=g_conv_dim,
                  num_gpu=num_gpu)

    # init weights of models
    D.apply(init_weights)
    G.apply(init_weights)

    # restore model weights
    if d_model_restore is not None and os.path.exists(d_model_restore):
        D.load_state_dict(torch.load(d_model_restore))
    if g_model_restore is not None and os.path.exists(g_model_restore):
        G.load_state_dict(torch.load(g_model_restore))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        D.cuda()
        G.cuda()

    return D, G

def make_variable(tensor):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def denormalize(x):
    """Invert normalization, and then convert array into image."""
    out = x * dataset_std_value + dataset_mean_value
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed():
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
init_random_seed()

# init models
D, G = get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_gpu,
                  d_model_restore, g_model_restore)

# init optimizer
criterion = nn.BCELoss()
if torch.cuda.is_available():
    criterion.cuda()
d_optimizer = optim.Adam(
    D.parameters(), lr=d_learning_rate, betas=(beta1, beta2))
g_optimizer = optim.Adam(
    G.parameters(), lr=g_learning_rate, betas=(beta1, beta2))

###############
# 2. training #
###############
fixed_noise = make_variable(torch.randn(
    batch_size, z_dim, 1, 1).normal_(0, 1))

for epoch in range(num_epochs):
    for step, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = make_variable(images)
        real_labels = make_variable(torch.ones(batch_size))
        fake_labels = make_variable(torch.zeros(batch_size))

        ##############################
        # (1) training discriminator #
        ##############################
        for d_step in range(d_steps):
            d_optimizer.zero_grad()

            noise = make_variable(torch.randn(
                batch_size, z_dim, 1, 1).normal_(0, 1))

            d_pred_real = D(images)
            d_loss_real = criterion(d_pred_real, real_labels)

            fake_images = G(noise)
            # use detach to avoid bp through G and spped up inference
            d_pred_fake = D(fake_images.detach())
            d_loss_fake = criterion(d_pred_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

        ##########################
        # (2) training generator #
        ##########################
        for g_step in range(g_steps):
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            noise = make_variable(torch.randn(
                batch_size, z_dim, 1, 1).normal_(0, 1))

            fake_images = G(noise)
            d_pred_fake = D(fake_images)
            g_loss = criterion(d_pred_fake, real_labels)
            g_loss.backward()

            g_optimizer.step()

        ##################
        # (3) print info #
        ##################
        if ((step + 1) % log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}]:"
                  "d_loss={} g_loss={} D(x)={} D(G(z))={}"
                  .format(epoch + 1,
                          num_epochs,
                          step + 1,
                          len(data_loader),
                          d_loss.data[0],
                          g_loss.data[0],
                          d_loss_real.data[0],
                          d_loss_fake.data[0]))

        ########################
        # (4) save fake images #
        ########################
        if ((step + 1) % sample_step == 0):
            if not os.path.exists(data_root):
                os.makedirs(data_root)
            fake_images = G(fixed_noise)
            torchvision.utils.save_image(denormalize(fake_images.data),
                                         os.path.join(
                                             data_root,
                                             "DCGAN-fake-{}-{}.png"
                                             .format(epoch + 1, step + 1))
                                         )

    #############################
    # (5) save model parameters #
    #############################
    if ((epoch + 1) % save_step == 0):
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        torch.save(D.state_dict(), os.path.join(
            model_root, "DCGAN-discriminator-{}.pkl".format(epoch + 1)))
        torch.save(G.state_dict(), os.path.join(
            model_root, "DCGAN-generator-{}.pkl".format(epoch + 1)))
