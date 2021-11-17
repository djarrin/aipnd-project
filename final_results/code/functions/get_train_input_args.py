import argparse

def get_train_input_args():
    parser = argparse.ArgumentParser()
    
    #get the data_dir
    parser.add_argument('data_dir', type=str, help='The directory that holds training, validation and test data')
    
    #save directory: --save_dir, default: checkpoint.pth
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'path to the folder the model checkpoint')

    #Architecture: --arch, default: alexnet
    parser.add_argument('--arch', type = str, default = 'alexnet', help = 'The name of the pretrained model to use, the allowed models are alexnet and vgg19')
    
    #cpu usage: --gpu
    parser.add_argument('--gpu', action='store_true', help = 'Will make use of a gpu to train network')
    #hyperparameters
    #learning rate: --learning_rate, default: 0.001
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'This will be the training learning rate')
    #hidden units: --hidden_units, default: 512
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'This will be the number of hidden layers')
    #epochs: --epochs, default: 20
    parser.add_argument('--epochs', type = int, default = 20, help = 'This will be the epochs the model will train for')
    
    return parser.parse_args()