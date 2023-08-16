
# John Lambert

import torch
import numpy as np
import sys
sys.path.append('../..')



def loss_fn_of_xstar(model, images_v, xstar_v, labels_v, opt, train,  criterion):
    x_output_v, sigmas = model(images_v, xstar_v, train )
    loss_x_v = criterion(x_output_v, labels_v)
    loss_v = loss_x_v

    if sigmas is not None:
        sigmas = torch.cat([sigmas[0].unsqueeze(0), sigmas[1].unsqueeze(0)], 0)
        sigmas_norm = torch.norm(sigmas, 2)
        loss_v = loss_v + opt.sigma_regularization_multiplier * sigmas_norm
    return loss_v, x_output_v

def loss_info_dropout(model, images_v, labels_v, train, criterion):
    """
    Criterion is Softmax-CE loss
    We add to cost function regularization on the noise (alphas) via the KL terms

    In all experiments we divide the KLdivergence term by the number of training
    samples, so that for beta = 1 the scaling of the KL-divergence term in similar to
    the one used by Variational Dropout
    """
    x_output_v, kl_terms = model(images_v, train)

    loss_v = None
    if train:
        kl_terms = [ kl.sum(dim=1).mean() for kl in kl_terms]
        if not kl_terms:
            kl_terms = [torch.tensor(0.)]
        N = images_v.size(0)
        Lz = (kl_terms[0] + kl_terms[1]) * 1. / N # sum the list

        Lx = criterion(x_output_v, labels_v)

        beta = 3.0
        if np.random.randint(0, 100) < 1:
            print(f'     [KL loss term: {beta * Lz.data[0]}')
            print(f'     [CE loss term: {Lx.data[0]}')

        loss_v = Lx + beta * Lz # PyTorch implicitly includes weight_decay * L2 in the loss
    return loss_v, x_output_v





