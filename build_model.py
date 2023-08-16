# John Lambert
from vgg_dlupi_model import DualNetworksVGG
from information_dropout import VGG_InformationDropout
from random_gauss_dropout_vgg import vgg16_bn_random_gaussian_dropout
from model_types import ModelType

def build_model(opt):
    """ Return the model on the GPU """
    print(opt.model_type)
    if opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
        print('We are using DualNetworks VGG for DLUPI')
        return DualNetworksVGG(opt)
    elif opt.model_type == ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE:
        # noise in train, but not in test...
        return vgg16_bn_random_gaussian_dropout()
    elif opt.model_type == ModelType.DROPOUT_INFORMATION:
        return VGG_InformationDropout(opt)
    else:
        print('undefined model type. quitting...')
        quit()