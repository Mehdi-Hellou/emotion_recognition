import os 
import numpy as np
from tensorflow.keras.utils import to_categorical

# Inspired from github files dataloader
# https://github.com/amineHorseman/facial-expression-recognition-using-cnn/blob/master/data_loader.py
path_training = "/home/mehdi/Mehdi/UoB/fer2013/data/Training"
path_test = "/home/mehdi/Mehdi/UoB/fer2013/data/PublicTest"
path_validation = "/home/mehdi/Mehdi/UoB/fer2013/data/PrivateTest"

def load_data(test=False, validation=False): 
	data_dict = {}

	data_dict['X'] = np.load(path_training+ '/images.npy')
	data_dict['X'] = data_dict['X'].reshape([-1, 48, 48, 1])

	data_dict['X2'] = np.load(path_training+ '/landmarks.npy')

	data_dict['Y'] = np.load(path_training+ '/labels.npy')   # one hot encoding 

	data_dict['Y'] = to_categorical(data_dict['Y'])
	if test:
		test_dict = {}
		test_dict['X'] = np.load(path_test+ '/images.npy')
		test_dict['X'] = test_dict['X'].reshape([-1, 48, 48, 1])

		test_dict['X2'] = np.load(path_test+ '/landmarks.npy')

		test_dict['Y'] = np.load(path_test+ '/labels.npy')
		test_dict['Y'] = to_categorical(test_dict['Y'])   # one hot encoding 

	if validation: 
		valid_dict= {}

		valid_dict['X'] = np.load(path_validation+ '/images.npy')
		valid_dict['X'] = valid_dict['X'].reshape([-1, 48, 48, 1])

		valid_dict['X2'] = np.load(path_validation+ '/landmarks.npy')

		valid_dict['Y'] = np.load(path_validation+ '/labels.npy')
		valid_dict['Y'] = to_categorical(valid_dict['Y'])  # one hot encoding 
	if not test and not validation: 
		return data_dict

	elif test and not validation:
		return data_dict,test_dict

	else: 
		return data_dict, test_dict, valid_dict 





if __name__ == '__main__':
	
	train, test, validation = load_data(True,True)

	
	X_train = train['X2']
	Y_train = train['Y']

	print(X_train.shape)
	print(Y_train.shape)