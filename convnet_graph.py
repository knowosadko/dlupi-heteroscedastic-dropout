
# John Lambert

import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchmetrics
from torch.utils.data.dataset import random_split
import lightning.pytorch as pl
import sys
import wandb
import sklearn
sys.path.append('..')

# ------------- Modules -------------------
from model_types import ModelType
from modular_loss_fns import loss_fn_of_xstar, loss_info_dropout
from pretrained_model_loading import load_pretrained_dlupi_model
from my_cifar import CLASSES
from build_model import build_model
from logger  import Logger

class ConvNet_Graph(pl.LightningModule):
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self._seed_config()
        self.current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Utilized device: {self.current_device}')
        if not os.path.isdir(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        self._model_setup()
        #self.logger_wrapper = Logger(self)
        self.criterion = nn.CrossEntropyLoss() # combines LogSoftMax and NLLLoss in one single class
        print(f'Using: {self.current_device}')
        #### REMOVE BELOW
        self.test_preds = None
        self.test_labels = None
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=10)
        self.training_step_number = 0
        self.validation_step_number = 0
        
    def _seed_config(self):
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        cudnn.benchmark = True

    def _log_image(self, images, labels, outputs, xstar):
        if xstar is None:
            image = images[0]
            tag = 'val'
        else:
            image = xstar[0]
            tag = 'train'           
        pred_label = CLASSES[torch.argmax(outputs[0])] 
        label = CLASSES[labels[0]]
        image_array = self.imsave(image)
        
        images = wandb.Image(
        image_array, 
        caption=f'Predicted class: {pred_label}. Actual class {label}.')      
        wandb.log({f"examples_{tag}": images})  
    
    
        
    def _my_log_roc(self, labels, outputs):
        wandb.log({"roc" : wandb.plot.roc_curve( labels.detach().cpu().numpy(), torch.argmax(outputs).detach().cpu().numpy(), \
                        labels=CLASSES, classes_to_plot=None)})

    
    def _model_setup(self):
        """ Virtual Function that can be replaced for a two-model convnet."""
        self.model = build_model(self.opt)
        self.model = self.model.to(self.current_device)
        print( list(self.model.modules()) )


    def configure_optimizers(self):# rename to configure optimizers
        print(f'We are using {self.opt.optimizer_type} with lr = {str(self.opt.learning_rate)}')
        if self.opt.optimizer_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), self.opt.learning_rate,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), self.opt.learning_rate,
                                         weight_decay=self.opt.weight_decay)
        else:
            raise Exception("Unknown optimizer.")

    def imsave(self, img): # move to utilities or sth
        img = img / 2 + 0.5     # unnormalize
        npimg = img.detach().cpu().numpy()
        sample = np.transpose(npimg, (1, 2, 0))
        if sample.shape[-1] < 2:
            zeros = np.zeros((sample.shape[0],sample.shape[1],2))
        else:
            zeros = np.zeros((sample.shape[0],sample.shape[1],1))
        sample = np.concatenate((sample,zeros),axis=-1)
        return sample

    def _save_model(self, avg_acc):
        if avg_acc > self.best_val_acc:
            torch.save({'state': self.model.state_dict(), 'acc': avg_acc}, os.path.join(self.opt.ckpt_path, 'model.pth'))
            self.best_val_acc = avg_acc

    def training_step(self, batch, batch_idx):# Add loggers
        images, labels, xstar = batch
        labels = torch.tensor(labels,  dtype=torch.long).to(self.current_device)
        images = torch.tensor(images, requires_grad=True, dtype=torch.float).to(self.current_device)
        xstar = torch.tensor(xstar, requires_grad=True, dtype=torch.float).to(self.current_device)
        loss, outputs = self._forward_pass(images, xstar, labels, True)
        self.log_train_step(images, labels, outputs, xstar,loss)
        return loss
  
    def validation_step(self, batch, batch_idx):# Add loggers
        images, labels, xstar = batch # should be images, labels = batch
        #images, labels = batch 
        xstar = None
        labels = torch.tensor(labels,  dtype=torch.long).to(self.current_device)
        images = torch.tensor(images, requires_grad=False, dtype=torch.float).to(self.current_device)
        loss, outputs = self._forward_pass(images, xstar, labels, False)
        self.log_val_step(images, labels, outputs, xstar,loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log_val_end()
        
    
    
    def test_step(self, batch, batch_idx):# Add loggers
        images, labels = batch
        xstar = None
        labels = torch.tensor(labels,  dtype=torch.long).to(self.current_device)
        images = torch.tensor(images, requires_grad=False, dtype=torch.float).to(self.current_device)
        xstar = None
        loss, outputs = self._forward_pass(images, xstar, labels, False)
        # self.log_test_step(labels,outputs)
        #self._log_confmat(labels, outputs)
        # self._log_roc(labels, outputs)
        return loss


    def _forward_pass(self, images_v, xstar_v, labels_v, train):
        """ Run model through a single feedforward pass.

        We return:
        -   loss_v: scalar loss value in the form of a PyTorch Variable
        -   x_output_v: logits in the form of Try running random gaussian dropout as wella PyTorch Variable
        """
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
            raise TypeError("Undefined model type. Quitting ...")
        
    def log_test_end(self):
        assert self.test_preds != None and self.test_labels != None
        #self._log_confmat(self.test_preds, self.test_labels)
        # self._log_roc(self.test_preds, self.test_labels)
    
    def log_train_step(self, images, labels, outputs, xstar,loss):
        self.log("train_loss", loss)
        if self.training_step_number % 10000:
            self._log_image(images, labels, outputs, xstar)
        self.training_step_number += 1
    
    def log_test_step(self,labels, outputs):
        if not self.test_labels and not self.test_preds:
            self.test_labels, self.test_preds = labels, torch.argmax(outputs,dim=1)
        torch.cat((self.test_labels.detach().cpu().numpy(),labels.detach().cpu().numpy()), dim=0)
        torch.cat((self.test_preds.detach().cpu().numpy(),torch.argmax(outputs,dim=1).detach().cpu().numpy()), dim=0)
    
    def log_val_end(self):
        self.log('valid_acc_epoch',self.valid_acc.compute())
        self.log('valid_f1_epoch', self.valid_f1.compute())
        self.valid_acc.reset()
        self.valid_f1.reset()
    
    def log_val_step(self,images, labels, outputs, xstar,loss):
        self.valid_acc.update(labels.detach().cpu(), torch.argmax(outputs, dim=1).detach().cpu())
        self.valid_f1.update(labels.detach().cpu(), torch.argmax(outputs, dim=1).detach().cpu())
        self.log("val_loss", loss)
        if self.validation_step_number % 100 == 0:
            self._log_image(images, labels, outputs, xstar)
        self.validation_step_number += 1
    
    def _log_confmat(self, labels, preds):
        cm = wandb.plot.confusion_matrix(
            y_true=labels.detach().cpu().numpy(),
            preds=preds.detach().cpu().numpy(),
            class_names=CLASSES)
        wandb.log({"conf_mat": cm})
        
    def _log_image(self, images, labels, outputs, xstar):
        if xstar is None:
            image = images[0]
            tag = 'val'
        else:
            image = xstar[0]
            tag = 'train'           
        pred_label = CLASSES[torch.argmax(outputs[0])] 
        label = CLASSES[labels[0]]
        image_array = self.imsave(image)
        images = wandb.Image(
        image_array, 
        caption=f'Predicted class: {pred_label}. Actual class {label}.')      
        wandb.log({f"examples_{tag}": images})
        
    def _log_roc(self, labels, preds):
        wandb.log({"roc" : wandb.plot.roc_curve( labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), \
                        labels=CLASSES, classes_to_plot=None)})
        
    def imsave(self, img): # move to utilities or sth
        img = img / 2 + 0.5     # unnormalize
        npimg = img.detach().cpu().numpy()
        sample = np.transpose(npimg, (1, 2, 0))
        if sample.shape[-1] < 2:
            zeros = np.zeros((sample.shape[0],sample.shape[1],2))
        else:
            zeros = np.zeros((sample.shape[0],sample.shape[1],1))
        sample = np.concatenate((sample,zeros),axis=-1)
        return sample
            

