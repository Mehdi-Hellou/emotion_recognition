import tensorflow as tf
import numpy as np
import pandas 
import dlib 
import cv2
import os
import imageio
from imutils import face_utils

image_height = 48
image_width = 48

path = '/media/mehdi/Donnés/Mehdi/ANDROIDE/Internship/UoB/dataset/616504_1103829_bundle_archive/icml_face_data.csv/icml_face_data.csv'

data = pandas.read_csv(path) 
data.columns = ['emotion', 'usage', 'pixels']  

path_images = "data/images/"
path_landmarks = "data/landmarks/"

# Load the detector of faces
detector = dlib.get_frontal_face_detector()
# Load the predictor of landmark
predictor = dlib.shape_predictor("../FACS/shape_predictor_68_face_landmarks.dat")

def get_landmarks(image, face_rects, rects):
	# this function have been copied from http://bit.ly/2cj7Fpq
	if len(face_rects) > 1:
		raise BaseException("TooManyFaces")
	if len(face_rects) == 0:
		print("Fails to detect face")
		shape = predictor(image, rects[0])
	else:
		shape = predictor(image, face_rects[0])
	return face_utils.shape_to_np(shape)

for category in data['usage'].unique():
    print( "converting set: " + category + "...")
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs(category)
        except OSError as e:
            print(e)
    
    # get samples and labels of the actual category
    category_data = data[data['usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values

    images = []
    landmarks = []
    labels_list = []
    for i in range(len(samples)): 

        # save images 
        image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
        image = image.astype(np.uint8)
        images.append(image)

        #imageio.imwrite(path_images + category + '/' + str(i) + '.jpg', image)

        #save landmarks
        imageio.imwrite("temp.jpg",image)

        image = cv2.imread('temp.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        face_rects = detector(gray, 1)
        rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(image, face_rects, rects)

        landmarks.append(face_landmarks)

        # get the labels 
        labels_list.append(labels[i])
        print("\n Processed image {} / {}".format(i,len(samples)))

    np.save(category + '/images.npy', images)
    np.save(category + '/landmarks.npy', landmarks)
    np.save(category + '/labels.npy', labels_list)
