# emotion_recognition

This github present a CNN training for emotions recognition based on the dataset fer2013. 

This dataset is available on kaggle (https://www.kaggle.com/deadskull7/fer2013) and consist of images from internet with different facial expressions. 
The images are labelled under 7 emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). 

# Models 

The models used are a basic Convolution Neural Network (CNN) combined with a feed-forward neural network. 

## CNN
The CNN is composed of 3 layers with the first layer including 64 masks(5X5), 128 masks(4X4) and 256 masks (5X5) respectively, 
followed by two fully connected layers of 1024 neurones and 40 neurones respectively. The CNN is applied to the images from the fer2013 dataset which are images of size 48*48. 

## Feed-forward neural network

The model is a simple feed-forward neural network(NN) of 4 layers. One input layer, a hidden layer (1024 neurones) and an output layer of 40 neurones. 

The feed-forward neural network is used to not learn directly on images but learn on landmark features extract from each image. Indeed, by using the file "fer_to_images_landmarks.py"
from "data" folder, we extract the landmark features of each image.

## Merging the two models
The output from the two models are merged and the final prediction, based on the 7 emotions, are made. 

# Files

## ConvNet.py

Model building and function to save it or change the learning rate during the training. 

## train.py
Training file 

## data_plot.py
File to plot the data.

## data folder  
### fer_to_images_landmarks.py

File to read the csv file from kaggle and extract the images and associated landmark from the fer2013 dataset. 

### data_loader.py

File to load the data when landmarks and image are extracted.  
