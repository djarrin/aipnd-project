from torch import nn
from torch import optim
from torchvision import models

def assemble_model(arch, hidden_units, learning_rate):
    status = True
    error_code = None
    
    if(arch == 'alexnet'):
        feature_input_size = 9216
        feature_output_size = 4096
        model_output_size = 1000
        model = models.alexnet(pretrained=True)
    elif(arch == 'vgg19'):
        feature_input_size = 25088
        feature_output_size = 4096
        model_output_size = 1000
        model = models.vgg19(pretrained=True)
    else:
        status = False
        error_code = 1
        return None, status, error_code
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #assemble classifier params based on the number of hidden layers the user specifies    
    params = nn.ModuleList([nn.Linear(feature_input_size, feature_output_size), nn.ReLU(), nn.Dropout(0.7)])
    for i in range(hidden_units):
        params.append(nn.Linear(feature_output_size, feature_output_size))
        params.append(nn.ReLU())
    params.append(nn.Linear(feature_output_size, model_output_size))
    params.append(nn.LogSoftmax(dim=1))
    
    model.classifier = nn.Sequential(*params)  
    
    model.criterion = nn.NLLLoss()
    model.optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, status, error_code