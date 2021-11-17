import torch
from torch import optim

def load_model(filepath):
    checkpoint = torch.load(filepath)

    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optomizer_state']
    model.epoch = checkpoint['epoch_number']
    model.class_to_idx = checkpoint['class_to_idx']
    model.criterion = checkpoint['criterion']
    
    return model

   