
# John Lambert

import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision

import sys
sys.path.append('..')

# ------------- Modules -------------------
from model_types import ModelType
from modular_loss_fns import loss_fn_of_xstar, loss_info_dropout
from pretrained_model_loading import load_pretrained_dlupi_model

from build_model import build_model
from my_cifar import MyCIFAR10

class ConvNet_Graph(object):
    
    def __init__(self, opt):
        self.opt = opt
        self._seed_config()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f'Utilized device: {self.device}')
        if not os.path.isdir(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        self.train_loader = self._build_dataloader_cifar(split='train')
        self.val_loader = self._build_dataloader_cifar( split='val')
        self._model_setup()

        self.criterion = nn.CrossEntropyLoss() # combines LogSoftMax and NLLLoss in one single class

        self.epoch = opt.start_epoch
        self.avg_val_acc = 0
        self.best_val_acc = -1  # initialize less than avg_acc
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.num_examples_seen_in_epoch = 0
        self.is_nan = False
        self.num_epochs_no_acc_improv = 0
        print(f'Using: {self.device}')
        
    def _seed_config(self):
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        #torch.cuda.manual_seed(1)
        cudnn.benchmark = True

    def _model_setup(self):
        """ Virtual Function that can be replaced for a two-model convnet."""
        self.model = build_model(self.opt)
        self.model = self.model.to(self.device)
        if self.opt.parallelize:
            self.model = torch.nn.DataParallel(self.model)
        if self.opt.model_fpath != '':
            if self.opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
                print('loading pre-trained model...')
                self.model = load_pretrained_dlupi_model(self.model, self.opt)
            else:
                self.model.load_state_dict(torch.load(self.opt.model_fpath)['state'])

        print( list(self.model.modules()) )
        self.optimizer = self._get_optimizer(self.model.parameters())

    def _build_dataloader_cifar(self, split='train' ):
        shuffle_data = True if (split == 'train') else False
        train = True if (split == 'train') else False
        dataset = MyCIFAR10(train=train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size,
                                                shuffle=shuffle_data, num_workers=2,pin_memory=True)
        print(f'train: {str(train)} loader has length {len(dataloader)}')
        print(f'{split} loader complete')
        return dataloader


    def _get_optimizer(self, model_parameters):
        print(f'We are using {self.opt.optimizer_type} with lr = {str(self.opt.learning_rate)}')
        if self.opt.optimizer_type == 'sgd':
            return torch.optim.SGD(model_parameters, self.opt.learning_rate,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer_type == 'adam':
            return torch.optim.Adam(model_parameters, self.opt.learning_rate,
                                         weight_decay=self.opt.weight_decay)
        else:
            print('undefined optim')
            quit()

    def _train(self):
        for epoch in range(self.opt.start_epoch, self.opt.num_epochs,1):
            self.epoch = epoch
            self._adjust_learning_rate()
            _ = self._run_epoch(tag='Train')
            if self.is_nan:
                print('loss is NaN')
                return False
            self.avg_val_acc  = self._run_epoch(tag='Val')
            print(f'Avg acc = {self.avg_val_acc} best_val_acc = {self.best_val_acc}')


    def _adjust_learning_rate(self):
        if self.opt.fixed_lr_schedule:
            self._decay_lr_fixed()
        else:
            self._decay_lr_adaptive()

    def _decay_lr_fixed(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.1 ** (self.epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _decay_lr_adaptive(self):
        """ Decay the learning rate by 10x if we cannot improve val. acc. over 5 epochs. """
        if self.avg_val_acc < self.best_val_acc:
            print(f'Avg acc = {self.avg_val_acc} not better than best val acc = {self.best_val_acc}')
            self.num_epochs_no_acc_improv += 1  # 1 strike -- learning rate needs to be decayed soon
        else:
            self.num_epochs_no_acc_improv = 0  # reset the counter
        if self.num_epochs_no_acc_improv >= self.opt.num_epochs_tolerate_no_acc_improv:  #
            for param_group in self.optimizer.param_groups:
                print(f'Learning rate was:{param_group["lr"]}')
                param_group['lr'] *= 0.1
                print(f'Learning rate decayed to:{param_group["lr"]}')
                self.num_epochs_no_acc_improv = 0  # reset the counter


    def _run_epoch(self, tag):
        """ Reset losses and accuracies for this new epoch. """
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.num_examples_seen_in_epoch = 0

        if tag == 'Val':
            train = False
            self._configure_model_state(train)
            split_data_loader = self.val_loader
        elif tag == 'Train':
            train = True
            self._configure_model_state(train)
            split_data_loader = self.train_loader

        for step, data in enumerate(split_data_loader):
            self._run_iteration(data, train, step, split_data_loader, tag)
            if self.is_nan:
                return -1

        avg_loss = np.sum(np.array(self.epoch_losses)) * (1.0 / self.num_examples_seen_in_epoch)
        avg_acc = np.sum(np.array(self.epoch_accuracies)) * (1.0 / self.num_examples_seen_in_epoch)
        print(f'[{tag}] epoch {self.epoch}/{self.opt.num_epochs}, step {step}/{len(split_data_loader) - 1}: avg loss={avg_loss:.4f}, avg acc={avg_acc:.4f}')

        if tag == 'Val':
            self._save_model(avg_acc)

        return avg_acc

    def _configure_model_state(self, train):
        if train == False:
            self.model.eval()
        else:
            self.model.train()

    def _save_model(self, avg_acc):
        if avg_acc > self.best_val_acc:
            torch.save({'state': self.model.state_dict(), 'acc': avg_acc}, os.path.join(self.opt.ckpt_path, 'model.pth'))
            self.best_val_acc = avg_acc

    def _run_iteration(self, data, train, step, data_loader, tag ):
        """ """
        if tag in ['Train' ]:
            images_t, labels_t, xstar_t = data
        elif tag == 'Val':
            images_t, labels_t = data
            xstar_t = None
        batch_size = images_t.size(0)
        self.num_examples_seen_in_epoch += batch_size

        labels_v = torch.tensor(labels_t,  dtype=torch.long).to(self.device)
        loss_v, x_output_v = self._forward_pass(images_t, xstar_t, labels_v, batch_size, train)

        preds_t = x_output_v.data.max(1)[1]
        accuracy = preds_t.eq(labels_v.data).cpu().sum() * 1. / ( batch_size * 1. )

        self.epoch_accuracies.append(accuracy * 1. * batch_size)
        if train:
            self.epoch_losses.append(loss_v.data.item() * 1. * batch_size)
            if np.isnan(float(loss_v.data.item())):
                self.is_nan = True

        if self.opt.print_every > 0 and step % self.opt.print_every == 0:
            if train:
                print(f'    [{tag} epoch  {self.epoch}/{self.opt.num_epochs}, step {step}/{len(data_loader) - 1}: loss={loss_v.data.item():.4f}, acc={accuracy:.4f}')
            else:
                print(f'    [{tag} epoch  {self.epoch}/{self.opt.num_epochs}, step {step}/{len(data_loader) - 1}: loss={loss_v.data.item():.4f}, acc={accuracy:.4f}')

        if train:
            self.optimizer.zero_grad()
            loss_v.backward()
            self.optimizer.step()


    def _forward_pass(self, images_t, xstar_t, labels_v, batch_size, train):
        """ Run model through a single feedforward pass.

        We return:
        -   loss_v: scalar loss value in the form of a PyTorch Variable
        -   x_output_v: logits in the form of Try running random gaussian dropout as wella PyTorch Variable
        """
        images_v = torch.tensor(images_t, requires_grad=train, dtype=torch.float).to(self.device)
        if train:
            xstar_v = torch.tensor(xstar_t, requires_grad=train, dtype=torch.float).to(self.device)
        else:
            xstar_v = None

        if self.opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
            return loss_fn_of_xstar(self.model, images_v, xstar_v, labels_v,
                                                  self.opt, train, self.criterion)

        if self.opt.model_type == ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE:  # NO x_star
            x_output_v = self.model(images_v, train)
            loss_v = self.criterion(x_output_v, labels_v)
            return loss_v, x_output_v

        if self.opt.model_type == ModelType.DROPOUT_INFORMATION:
            return loss_info_dropout(self.model, images_v, labels_v, train, self.criterion)
        else:
            print('Undefined model type. Quitting...')
            quit()

