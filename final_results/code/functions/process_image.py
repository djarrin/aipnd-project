from PIL import Image
import numpy as np
import torch

def process_image(image):
    #open image
    im = Image.open(image)
    
    #resize image
    size = 256, 256
    im.thumbnail(size)
    
    # resize so that the shortest side is 256
    if im.size[0] > im.size[1]:
        im.thumbnail((500000, 256))
    else:
        im.thumbnail((256, 500000))
    # center crop the image 
    left_margin = (im.width-224)/2
    bottom_margin = (im.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    im = im.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
        
    #divide image by 255 to convert the color channel
    np_image = np.array(im)
    im = np_image/255
    
    #normalize color channels
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    im = im - means
    im = im/stds
    
    #re-arrange elements in the array
    im = im.transpose((2,0,1))
    
    tensor_version = torch.from_numpy(im)
    return tensor_version