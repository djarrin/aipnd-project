import sys
from functions.get_train_input_args import get_train_input_args
from functions.load_data import load_data
from functions.assemble_model import assemble_model
from functions.train_model import train_model
from functions.save_model_checkpoint import save_model_checkpoint
import torch

def main():
    #get input arguements
    in_args = get_train_input_args()
    if(len(in_args.save_dir.split('.')) == 1 or in_args.save_dir.split('.')[1] != 'pth'):
        print('save_dir must have .pth file exstension')
        return
        
    #get dataloaders and datasets
    image_datasets, dataloaders, dataset_sizes = load_data(in_args.data_dir)
    
    model, status, error_code = assemble_model(in_args.arch, in_args.hidden_units, in_args.learning_rate)
    if(status == False and error_code == 1):
        print('You must choose either alexnet or vgg19')
        return
    elif(status == False):
        print('something went wrong with model creation')
        return
    
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.gpu) else "cpu")
        
    model.to(device)
    
    model = train_model(model, in_args.epochs, device, dataloaders)
    
    save_model_checkpoint(model, in_args.save_dir, model.state_dict(), in_args.epochs, image_datasets['train'].class_to_idx, in_args.arch)

if __name__ == "__main__":
    main()
