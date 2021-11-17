import torch
from torch import nn

def save_model_checkpoint(model, save_dir, state_dict, epochs, class_to_idx, arch):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'mType': arch,
        'state_dict': state_dict,
        'epoch_number': epochs,
        'optomizer_state': model.optimizer.state_dict,
        'model': model,
        'criterion': nn.NLLLoss(),
        'class_to_idx': model.class_to_idx,
    }
    
    #important to note that if there is not enough disk space this will error out, if errors occur delete other saved models
    torch.save(checkpoint, save_dir)