import sys

sys.path.insert(1, 'data/')

import os 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from data_loader import load_data

IMAGE_SIZE = (139,139)

def augment(inputs, label):
    image = inputs['img_input']
    landmark = inputs['ldmk_input']

    image = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(image)    
    landmark = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(landmark) 
    image = tf.image.resize(image,IMAGE_SIZE)  # resize the image
    
    nb_rot =  tf.random.uniform([1,1],maxval = 10, dtype=tf.int32)
    image = tf.image.rot90( image, k=int(nb_rot[0][0]), name=None)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    inputs['img_input'] = image
    inputs['ldmk_input'] = landmark
    return inputs, label   

class ConvNet(tf.keras.Model):
    """docstring for ConvNet"""
    """ Model from the paper Kim et al, 2016"""
    def __init__(self,num_classes,IMG_SHAPE):
        super(ConvNet,self).__init__()
        
        self.conv1 = tf.keras.layers.Convolution2D(64, (5, 5), activation='relu')
        self.max1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.conv2 = tf.keras.layers.Convolution2D(128, (4, 4), activation='relu')
        self.max2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.conv3 = tf.keras.layers.Convolution2D(256, (5, 5), activation='relu')
        self.max3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = tf.keras.layers.Flatten()   # add after

        """self.d1 = tf.keras.layers.Dense(4096,activation ='relu', name="d1")
        self.drop1 = tf.keras.layers.Dropout(0.2)

        self.d2 = tf.keras.layers.Dense(1024,activation ='relu', name="d2")  # second saved model 
        self.drop2 = tf.keras.layers.Dropout(0.5)"""

        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.d1_c = tf.keras.layers.Dense(1024,activation ='relu', name="d1_convolution")  # first saved model       
        self.d2_c = tf.keras.layers.Dense(40,activation ='relu', name="d2_convolution")

        self.d1_l = tf.keras.layers.Dense(1024,activation ='relu', name="d1_landmarks")  # first saved model
        self.d2_l = tf.keras.layers.Dense(40,activation ='relu', name="d2_landmarks")  # first saved model

        self.out = tf.keras.layers.Dense(num_classes,activation = 'softmax',name = "output")
    
    def call(self,inputs):
        image = inputs['img_input']   # images inputs 
        landmark = inputs['ldmk_input'] # landmarks inputs  
        ### Convolutions neural network##########
        conv1 = self.conv1(image)
        max1 = self.max1(conv1)

        conv2 = self.conv2(max1)
        max2 = self.max2(conv2)

        conv3 = self.conv3(max2)
        max3 = self.max3(conv3)
        
        flatten = self.flatten(max3)

        drop1 = self.drop1(flatten)
        d1_c = self.d1_c(drop1)
        d2_c = self.d2_c(d1_c)
        
        """d2 = self.d2(drop1)
        drop2 = self.drop2(d2)"""
        
        ########################
        ### landmark detectors ##########
        flatten = self.flatten(landmark)  # get one vector
        d1_l = self.d1_l(flatten)
        d2_l = self.d2_l(d1_l) 
        #### output ###########
        merge = tf.keras.layers.concatenate([d2_c,d2_l], name = "merge")  # merge the features from the landmarks and the convolutions operations 
        
        output = self.out(merge)

        return output

class EarlyStopping():
    """docstring for EarlStopping"""
    def __init__(self, epoch_treshold):
        self.best_value_acc = 0.0
        self.epoch_treshold = epoch_treshold
        self.nb_epoch = 0

    def update(self, acc): 
        
        if self.best_value_acc < acc: 
            self.best_value_acc = acc
            self.nb_epoch = 0
        else: 
            self.nb_epoch+=1

        if self.nb_epoch == self.epoch_treshold: 
            return True
        else: 
            return False

class Scheduler():
    """docstring for Scheduler"""
    def __init__(self, epoch_treshold):
        self.best_accuracy = 0.0  # variable to check if the accuracy is changing over the epochs
        self.epoch_treshold = epoch_treshold # epoch treshold to change the learning rate
        self.nb_epoch = 0  

    def update(self,accuracy,optimizer):
        updated_lr = optimizer.learning_rate / 10
        print("\nbest acc : {} ; acc : {}; nb epoch: {},"
                .format(self.best_accuracy,accuracy,self.nb_epoch))

        if updated_lr < 1e-6: # a treshold for the learning rate to not have one with an extreme low values  
            updated_lr = 1e-6

        if self.best_accuracy > accuracy: # if the accuracy doesn't improve in the current epoch according to the prievous one 
            self.nb_epoch += 1
        else:                  # if the accuracy improves in the current epoch according to the prievous one        
            self.best_accuracy = accuracy   
            self.nb_epoch = 0

        if self.nb_epoch==self.epoch_treshold:      # we update the learning rate if the number of epoch 
                                                    # where the accuracy has not improved is over two epochs
            self.nb_epoch = 0  
            return updated_lr
        else:
            return optimizer.learning_rate
        
class SaveModel():
    """docstring for SaveModel"""
    def __init__(self):
        self.best_accuracy = 0.0  # variable to check if the accuracy is changing over the epochs

    def save_update(self,model, accuracy, optimizer):
        if self.best_accuracy < accuracy: # if the accuracy improves in the current epoch according to the prievous one
            model.save_weights("trained_model/model_1")
            self.best_accuracy = accuracy   
            print("\n saved model with best accuracy : {}."
                .format(self.best_accuracy))       
                              
if __name__ == '__main__':

    IMG_SHAPE = IMAGE_SIZE + (1,)
    #my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    #tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    
    model = ConvNet(7,IMG_SHAPE)
    train, test, validation = load_data(True,True)

    X_test, X_test2, Y_test = test['X'], test['X2'], test['Y'] 

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {"img_input": X_test, "ldmk_input": X_test2},
        Y_test,
    ))
    test_dataset = test_dataset.shuffle(buffer_size=1024)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_dataset = (
    test_dataset
    # The augmentation is added here.
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
    )

    for inputs, target in test_dataset:
        
        image = inputs['img_input'][0]
        
        plt.imshow(image[:,:,0], cmap='gray',interpolation='none')
        plt.show()

        output = model(inputs)

        print(output)
        print(model.summary())
        break;