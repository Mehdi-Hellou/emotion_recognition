import sys

sys.path.insert(1, 'data/')

import tensorflow as tf 
import matplotlib.pyplot as plt

import numpy as np 

from data_plot import *
from ConvNet import *

from signal import signal, SIGINT
from sys import exit

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
emotion_class = ["Angry","Disgust","Fear","Happy","Sad", "Surprise", "Neutral"]

loss_obj = tf.keras.losses.CategoricalCrossentropy()   # loss function for one hot 
#loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()   # loss function()   # loss function for not one hot 
optimizer = tf.keras.optimizers.Adam()  # the optimizer for the training
#ptimizer = tf.keras.optimizers.SGD(0.1)  # the optimizer for the training 

train_loss = tf.keras.metrics.Mean(name = "train_loss")
test_loss = tf.keras.metrics.Mean(name = "test_loss")
valid_loss = tf.keras.metrics.Mean(name = "valid_loss")
#accuracy
train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_acc = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
valid_acc = tf.keras.metrics.CategoricalAccuracy(name="valid_accuracy")

IMAGE_SIZE = (139,139)

# neural network model 
IMG_SHAPE = IMAGE_SIZE + (1,)
model = ConvNet(7,IMG_SHAPE)
model.compile(optimizer = optimizer,
             loss = "categorical_crossentropy",
              metrics = ["categorical_accuracy"])

checkpoint = tf.train.Checkpoint(step=tf.Variable(1),optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(
    checkpoint, directory="saved_model/", max_to_keep=2)

def resize_rescale(inputs,label): 
    """
    preprocessing for test images 
    """
    image = inputs['img_input']
    landmark = inputs['ldmk_input']

    image = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(image)    
    landmark = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(landmark)
    image = tf.image.resize(image,IMAGE_SIZE)  # resize the image to the size 197*197

    inputs['img_input'] = image
    inputs['ldmk_input'] = landmark
    return inputs, label

def augment(inputs, label):
    """
    preprocessing for test images 
    """
    image = inputs['img_input']
    landmark = inputs['ldmk_input']

    image = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(image)    
    landmark = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)(landmark)  
    image = tf.image.resize(image,IMAGE_SIZE)  # resize the image to the size 197*197

    nb_rot =  tf.random.uniform([1,1],maxval = 10, dtype=tf.int32)
    image = tf.image.rot90( image, k=int(nb_rot[0][0]), name=None)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    inputs['img_input'] = image
    inputs['ldmk_input'] = landmark
    return inputs, label

@tf.function
def train_step(inputs,targets): 
    global loss_obj,train_loss,train_acc, model
    
    with tf.GradientTape() as tape: 
        predictions = model(inputs)
        loss = loss_obj(targets,predictions)
    
    gradients = tape.gradient(loss,model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))  
    train_loss(loss)
    train_acc(targets,predictions)

@tf.function
def test_step(inputs,targets):
    global loss_obj,test_loss, test_acc, model 

    predictions = model(inputs)
    t_loss = loss_obj(targets,predictions)
    
    test_loss(t_loss)
    test_acc(targets,predictions)

@tf.function
def valid_step(inputs, targets):
    global loss_obj,valid_loss, valid_acc, model 

    predictions = model(inputs)
    t_loss = loss_obj(targets,predictions)
    
    valid_loss(t_loss)
    valid_acc(targets,predictions)

def main(batch_size,epochs):
    global test_acc, test_loss, train_acc, train_loss, valid_acc, valid_loss

    es = EarlyStopping(7)
    scheduler = Scheduler(5)
    saved = SaveModel()

    train, test, valid = load_data(True,True)

    X_train, X_train2, Y_train = train['X'], train['X2'], train['Y']
    X_test, X_test2, Y_test = test['X'], test['X2'], test['Y'] 
    X_valid, X_valid2, Y_valid = valid['X'], valid['X2'], valid['Y']

    X_test,X_test2 = np.concatenate((X_test, X_valid)),np.concatenate((X_test2, X_valid2))  # image and landmark 
    Y_test = np.concatenate((Y_test, Y_valid))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {"img_input": X_test, "ldmk_input": X_test2},
        Y_test,
    ))
    test_dataset = test_dataset.shuffle(buffer_size=1024)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"img_input": X_train, "ldmk_input": X_train2},
            Y_train,
        )
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = (
    train_dataset
    # The augmentation is added here.
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
    ) 

    test_dataset = (
    test_dataset
    # The augmentation is added here.
    .map(resize_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
    ) 

    acc_loss_plot = AccLossPlot("loss_values/","accuracy_values/")

    acc_loss_plot.load_value()
    
    # restore checkpoint if exist 
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # load previous weight if one
    start_epoch = int(checkpoint.step)
    b =  start_epoch

    for epoch in range(start_epoch,epochs):

        template = '\n LR: {}, epochs: {}'
        print(template.format(optimizer.learning_rate,epoch))
        
        for inputs, targets_batch in train_dataset:
            # Train step
            train_step(inputs, targets_batch)
            template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
            
            print(template.format(
                b, len(Y_train), train_loss.result(), 
                train_acc.result()*100
            ), end="")
            b += batch_size
            
        # Test set
        for inputs, targets_batch in test_dataset:
            test_step(inputs, targets_batch)
            
        template = '\nEpoch {}, test Loss: {}, test Accuracy: {}'
        print(template.format(
            epoch,
            test_loss.result(), 
            test_acc.result()*100)
        )

        acc_loss_plot.update(train_loss.result(), test_loss.result(), train_acc.result(), test_acc.result())

        # early stop 
        model.stop_training = es.update(test_acc.result()*100)
        optimizer.learning_rate = scheduler.update(test_acc.result()*100,optimizer) # change the learning rate or not according to the situation
        saved.save_update(model, test_acc.result()*100,optimizer)  # try to save the best model in the folder fer2013/trained_model

        test_loss.reset_states()
        test_acc.reset_states()
        train_acc.reset_states()
        train_loss.reset_states() 

        checkpoint.step.assign_add(1)
        # saved the model 
        if int(checkpoint.step) % 5 == 0 and int(checkpoint.step)!=0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            acc_loss_plot.save_value()  # save loss and accuracy values in json files

if __name__ == '__main__':    
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    main(128,100)