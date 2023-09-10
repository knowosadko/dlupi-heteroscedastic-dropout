
# John Lambert
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import collections

from reparameterization_trick import sample_lognormal

import sys
sys.path.append('..')

import os
import numpy as np
import pdb

# Gaussian Dropout as a function of x
class VGG_InformationDropout(nn.Module):
    def __init__(self, opt):
        super(VGG_InformationDropout, self).__init__()
        self.opt = opt
        self.FC_size = self.opt.fc_size

        self.lognorm_prior = False
        self.max_alpha = 0.7
        self.activation_fn = 'relu' # 'sigmoid' # not 'softplus'

        # CONV LAYERS DO NOT REQUIRE DROPOUT
        self.cfg =  [64, 64, 'M',
                     128, 128, 'M',
                     256, 256, 256, 'M',
                     512, 512, 512, 'M',
                     512, 512, 512, 'M']

        self.features = self.make_layers()

        # If learning these params fails, can also try:
        # mu1 = 0.8 * torch.ones(alphas[0].size()).type(torch.cuda.FloatTensor)
        # sigm1 = 0.8build_mo

        # mu1 = 0.5
        # sigma1 = 0.4

        self.mu1 = nn.Parameter(torch.rand(1)) # learned scalar, requires grad by default
        self.sigma1 = nn.Parameter(torch.rand(1)) # learned scalar, requires grad by default

        out_recept_fld_sz = opt.image_size / (2 ** 5 ) # 5 max pools that shrink size

        assert out_recept_fld_sz == 7
        assert self.FC_size == 4096

        flattened_feat_sz = 512 * out_recept_fld_sz * out_recept_fld_sz
        flattened_feat_sz = int(flattened_feat_sz) # Int must be Tensor Size input

        # Using self.opt.activation_fn == 'relu'
        # Could also use self.opt.activation_fn == 'softplus' ; x = self.softplus(x)

        self.x_fc1 = nn.Sequential(nn.Linear(flattened_feat_sz, self.FC_size), nn.Sigmoid() )
        self.fc1_alpha = nn.Sequential(nn.Linear(flattened_feat_sz, self.FC_size), nn.Sigmoid() )

        # Squash input here to prevent unbounded noise... (Sigmoid instead of nn.ReLU(True) )
        self.x_fc2 = nn.Sequential(nn.Linear(self.FC_size, self.FC_size), nn.Sigmoid() )
        self.fc2_alpha = nn.Sequential(nn.Linear(self.FC_size, self.FC_size), nn.Sigmoid() )

        assert self.opt.num_classes == 1000
        self.x_fc3 = nn.Linear(self.FC_size, self.opt.num_classes)

        self._initialize_weights()

    def make_layers(self, batch_norm=True):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    activation_fn = nn.ReLU(inplace=True) if self.activation_fn == 'relu' else (
                                    nn.Softplus() if self.activation_fn == 'softplus' else nn.Sigmoid())
                    layers += [conv2d, nn.BatchNorm2d(v), activation_fn]
                else:
                    activation_fn = nn.ReLU(inplace=True) if self.activation_fn == 'relu' else (
                                    nn.Softplus() if self.activation_fn == 'softplus' else nn.Sigmoid())
                    layers += [conv2d, activation_fn]

                in_channels = v
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def _KL_div2(self, mu, sigma, mu1, sigma1):
        '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
        return 0.5 * ((sigma / self.sigma1) ** 2 + (mu - self.mu1) ** 2 / self.sigma1 ** 2 - 1 + 2 * (torch.log(self.sigma1) - torch.log(sigma)))


    def _information_dropout(self, x_out, alpha, sigma0=1. ):
        """ We already computes the noise parameter alpha from its own FC layer based on the input"""

        # Rescale alpha in the allowed range and add a small value for numerical stability
        alpha = 0.001 + self.max_alpha * alpha
        # Similarly to variational dropout we renormalize so that
        # the KL term is zero for alpha == self.max_alpha
        if not self.lognorm_prior:
            kl = -1. * torch.log(alpha / (self.max_alpha + 0.001))
        else:
            # info dropout for softplus
            kl = self._KL_div2(torch.log(torch.max(x_out,1e-4)), alpha)

        zero_mean = torch.zeros(x_out.size(),dtype=torch.float)
        noise = sample_lognormal(mean=zero_mean, sigma=alpha, sigma0=sigma0)
        # Noisy output of Information Dropout
        return x_out * noise, kl


    def forward(self, x, train):
        if train:
            self.sigma0 = 1.
        else:
            self.sigma0 = 0 # will turn noise into exp(0), which =1

        x = self.features(x)
        x = x.view(x.size(0), -1)

        x_cloned = x.clone()
        x = self.x_fc1(x) # compute the noiseless output, includes relu
        alpha = self.fc1_alpha(x_cloned)

        if train:
            x, kl_term1 = self._information_dropout(x, alpha)

        x_cloned = x.clone()
        x = self.x_fc2(x) # compute the noiseless output, includes relu
        alpha = self.fc2_alpha(x_cloned)

        if train:
            x, kl_term2 = self._information_dropout(x, alpha)

        x = self.x_fc3(x)

        if train:
            kl_terms = [kl_term1, kl_term2]
        else:
            kl_terms = None

        return x, kl_terms