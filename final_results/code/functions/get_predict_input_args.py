import argparse

def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    #get image path
    parser.add_argument('image_path', type=str, help='The path to the image we would like to predict')
    
    #get checkpoint path
    parser.add_argument('checkpoint_path', type=str, help='The path to the saved model checkpoint')
    
    #cpu usage: --gpu
    parser.add_argument('--gpu', action='store_true', help = 'Will make use of a gpu to predict')
    
    #top_k: --top_k, default: 3
    parser.add_argument('--top_k', type = int, default = 3, help = 'determines how many images to show')
    
    #category names: --category_names , default: cat_to_name.json
    parser.add_argument('--category_names', type = str, default = 'none', help = 'gives the path to the json mapping file')
    
    return parser.parse_args()