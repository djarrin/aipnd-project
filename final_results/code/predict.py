from functions.get_predict_input_args import get_predict_input_args
from functions.load_model import load_model
from functions.process_image import process_image
import json
import torch

def main():
    #get input arguements
    in_args = get_predict_input_args()
    
    model = load_model(in_args.checkpoint_path)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.gpu) else "cpu")
    
    probs, classes, flowers = predict(in_args.image_path, model, device, in_args.category_names, in_args.top_k)
    
    if(flowers != None):
        print(f"Most likely flower is the {flowers[0]} or class {classes[0]} with a probablity of being right of {probs[0]}")
        print("The next top k results you requested are:")
        for i in range(len(probs)):
            if i != 0:
                print(f"{i + 1}. Flower is the {flowers[i]} or class {classes[i]} with a probablity of being right of {probs[i]}")
    else:
        print(f"Most likely class is {classes[0]} with a probablity of being right of {probs[0]}")
        print("The next top k results you requested are:")
        for i in range(len(probs)):
            if i != 0:
                print(f"{i + 1}. Class {classes[i]} with a probablity of being right of {probs[i]}")
    
    
def predict(image_path, model, device, category_names, topk=5):
    #process image
    im = process_image(image_path)

    inputs = im.to(device, dtype=torch.float)
   
    inputs = inputs.unsqueeze_(0)
    probs = torch.exp(model.forward(inputs)) 
    probs, classes = probs.topk(topk) 
    probs = probs.detach().cpu().numpy().tolist()[0] 
    classes = classes.detach().cpu().numpy().tolist()[0] 

    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    
    labels = [idx_to_class[class1] for class1 in classes]
    if(category_names != 'none'):
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        flowers = [cat_to_name[idx_to_class[class1]] for class1 in classes]
        return probs, classes, flowers
    
    return probs, classes, None
        
if __name__ == "__main__":
    main()