
# John Lambert

import torch
import pdb

def load_pretrained_model(model, opt):
    """ Load model weights from disk into the model that sits in main memory. """
    ckpt_dict = torch.load(opt.model_fpath)
    ckpt_state = ckpt_dict['state']
    print('loaded ckpt with accuracy:', ckpt_dict['acc'])
    model.load_state_dict(ckpt_state)
    return model

def load_pretrained_dlupi_model(model, opt):
    """
    Load model weights from disk into the model that sits in main memory.
    Exclude the loading of buffers.
    """
    model_fpath = opt.model_fpath
    saved_obj = torch.load(model_fpath)
    print('Loading model with accuracy:', saved_obj['acc'])
    saved_ckpt_dict = saved_obj['state']
    curr_model_state_dict = model.state_dict()

    updated_dict = {}
    # 1. filter out unnecessary keys
    for model_key in curr_model_state_dict.keys():
        if ('running_std_1' in model_key) or ('running_std_2' in model_key):
            print('Skipping loading of', model_key)
            continue
        print('     loaded weight for:', model_key, 'from', model_key)
        updated_dict[model_key] = saved_ckpt_dict[model_key]

    # 2. overwrite entries in the existing state dict
    curr_model_state_dict.update(updated_dict)
    # 3. load the new state dict
    model.load_state_dict(curr_model_state_dict)
    return model


