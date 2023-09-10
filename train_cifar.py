# John Lambert,  Ozan Sener
import argparse
import os.path
import torch
import datetime
from torch.utils.data.dataset import random_split
import lightning.pytorch as pl
from pytorch_lightning import loggers as pl_loggers
import sys
sys.path.append('../..')

from convnet_graph import ConvNet_Graph
from model_types import ModelType
from fixed_hyperparams import get_fixed_hyperparams
from my_cifar import MyCIFAR10
from logger  import Logger
def main(opt):
    """
    12/14 of our baseline models utilize a single computational graph.
    2/14 of our models (for curriculum learning) utilize two computational graphs.
    We instantiate the appropriate class (one is a derived class of the other).
    """
    convnet_graph = ConvNet_Graph(opt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    convnet_graph.to(device)
    train(convnet_graph)


def train(model): 
        batch_size = 16 # Remove it and fix it later
        # Create dataset
        train_data = MyCIFAR10(train=True)
        test_dataset = MyCIFAR10(train=False)
        # Split dataset into validation and train
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        # Gets data splits
        train_dataset, val_dataset = random_split(
                train_data,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
                )
        # Initilize dataloaders: trainloader, validation loader and test loader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2,pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2,pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2,pin_memory=True)
        # Initilize trainer
        # add loggers
        # Call trainer.fit()
        wb_logger = pl_loggers.WandbLogger("random _gaussian_noise/")
        wb_logger.watch(model.model)
        trainer = pl.Trainer(logger=wb_logger, max_epochs=250, log_every_n_steps=100, accelerator="gpu", devices=1)
        trainer.fit(model, train_dataloader, val_dataloader)
        # validation
        trainer.test(dataloaders=test_dataloader)
        model.log_test_end()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ---------------------- HYPERPARAMS PER EXPERIMENT-------------------------------------
    parser.add_argument('--trained_with_xstar', type=bool, default=True)
    parser.add_argument('--model_fpath', type=str,
                        default = '')
                        # 200k random gaussian dropout to resume
                        #default = '/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_27_14_50_57_num_ex_per_cls_200_bs_256_optimizer_type_sgd_model_type_ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE_lr_0.01_fixlrsched_False/model.pth' )

                        # 600k no x* model, we use this for curriculum learning
                        #default = '/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_09_00_25_00_num_ex_per_cls_600_bs_256_optimizer_type_sgd_dropout_type_bernoulli_lr_0.01_fixlrsched_False/model.pth')

    #parser.add_argument('--curric_phase_1_model_fpath', type=str, default = '')
                       # Phase 1 Complete
                       #default='/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_28_07_44_21_num_ex_per_cls_600_bs_128_optimizer_type_adam_model_type_ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1_lr_0.001_fixlrsched_False/epoch_7_learn_xstar_tower_model.pth')

    parser.add_argument('--start_epoch', type=int, default=0)

    #parser.add_argument('--dgx', type=bool, default=True)
    parser.add_argument('--fixed_lr_schedule', type=bool, default=False) # False means adaptive
    parser.add_argument('--batch_size', type=int, default= 32 )
    parser.add_argument('--optimizer_type', type=str, default='sgd') # otherwise, 'sgd'
    parser.add_argument('--learning_rate', type=float, default=1e-2 ) # 0.01 for sgd
    parser.add_argument('--model_type', type=str, default= ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE)

        # DROPOUT_FN_OF_XSTAR
        # DROPOUT_RANDOM_GAUSSIAN_NOISE


    parser.add_argument('--parallelize', type=bool, default=False) # Maybe should be removed
    parser = get_fixed_hyperparams(parser)
    opt = parser.parse_args()

    cur_datetime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    opt.ckpt_path = cur_datetime
    opt.ckpt_path += '_bs_'+ str(opt.batch_size)
    opt.ckpt_path += '_optimizer_type_' + str( opt.optimizer_type )
    opt.ckpt_path += '_model_type_' + str( opt.model_type )

    #opt.ckpt_path += '_percent_of_xstar_to_use_in_train_' + str( opt.percent_of_xstar_to_use_in_train )
    opt.ckpt_path += '_lr_' + str(opt.learning_rate)
    opt.ckpt_path += '_fixlrsched_' + str(opt.fixed_lr_schedule)

    save_dir = "home/konrad/..."
   
   
    opt.ckpt_path = os.path.join(save_dir, opt.ckpt_path)

    opt.num_classes = 10
    opt.num_channels = 1
    opt.image_size = 32

    opt.new_localization_train_path = os.path.join( opt.dataset_path, 'train')
    opt.new_localization_val_path = os.path.join( opt.dataset_path, 'val')
    #opt.new_localization_annotation_path = os.path.join( opt.dataset_path, 'localization_annotation')


    print('Parameters:', opt)
    opt.cuda = torch.cuda.is_available()

    main(opt)