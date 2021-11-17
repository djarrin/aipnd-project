# AI Programming with Python Project

This project allows for user to train either a alexnet or vgg19 model with any variation of hidden layers 
Train Inputs:
data_dir --required this is where data is housed for the model to train off of, inside this directory should be three directories named train/, valid/, test/
--save_dir this is where the checkpoint will be loaded save_dir must have a file exstension of .pth
--arch is the starting archetecture to be used (it can either be alexnet or vgg19)
--gpu the option to gpu rather than cpu
--learning_rate the intial learning rate at which the optimizer will correct
--hidden_units the number of hidden layers to add to the classifier
--epochs the number of epochs to train the model for

example command: python train.py flowers --gpu --hidden_units=5 --epochs=1 --save_dir="alexnet_gpu_var_hidden.pth"

Predict Inputs
image_path --required the path of the image for the model to guess
checkpoint_path --required the path of the model checkpoint to load
--gpu whether to use gpu for prediction or cpu
--top_k how many results to pull (the top 5 guesses along with their probabilities)
--category_names if included will include the data classified name, if this is included this should be the path to the mapping json file

example command: python predict.py "flowers/test/1/image_06743.jpg" --gpu --top_k=5 --category_names="cat_to_name.json" 

Special notes
If you run into issues with saving the model at the checkpoint after training please ensure you have enough disk space, this may include deleting other saved checkpoints
If you train the model with the GPU then you need to predict with the GPU, if you train with the CPU then you need to predict with the CPU

