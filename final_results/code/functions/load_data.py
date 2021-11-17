import torch
from torchvision import datasets, transforms

def load_data(data_dir):
    
    #define directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #define training transform
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomGrayscale(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

    #define test_validation transform
    test_validation_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 
    
    data_transforms = {
        'train': train_transform,
        'valid': test_validation_transform,
        'test': test_validation_transform
    }   

    dirs = {
        'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir
    }
    
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders, dataset_sizes
    