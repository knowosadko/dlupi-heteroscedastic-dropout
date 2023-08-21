
# John Lambert

import torch


def vanilla_reparametrize( mu, std ):
    eps = torch.randn(std.size()).to(std.device)
    return eps.mul(std).add_(mu), std


def vae_reparametrize( mu, logvar, distribution= 'normal'):
    std = logvar.mul(0.5).exp_()
    if distribution == 'normal':
        eps = torch.randn(std.size()).to(std.device)
    else:
        raise TypeError("undefined distribution for reparam trick. quitting...")
    return eps.mul(std).add_(mu), std



def sample_lognormal(mean, sigma, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)

    .normal() gives mean=0, std=1.
    """
    eps = torch.randn(mean.size()) #mean.data.new(mean.size()).normal_()
    return torch.exp(mean + sigma * sigma0 * eps )