import csv
import cv2
import numpy as np
import time

file_contains_header = True

dev_mode = False
no_of_images_to_read_in_dev_mode = 1000

# Read the data collected in csv file

samples = []

with open('../CarND-Behavioral-Cloning-P3-data/data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	first_line = True
	for sample in reader:
		if(first_line and file_contains_header):
			first_line = False
			continue
		samples.append(sample)
		if(dev_mode and len(samples) >= no_of_images_to_read_in_dev_mode):
			break
	
# Update the image path from absolute path to relative path as we plan to train the model in AWS and the absolute path will be different there



start_time = time.time()

# Split the data between training and validation set
from sklearn.model_selection import train_test_split
from random import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

image_shape = (160,320,3)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	relative_path = '../CarND-Behavioral-Cloning-P3-data/data/IMG'
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset+batch_size]
		
			images = []
			measurements = []
			
			for batch_sample in batch_samples:
				for i in range(3):
					image_file_name = batch_sample[i].split('/')[-1]
					relative_file_name = relative_path + '/' + image_file_name
					input_image = cv2.imread(relative_file_name)
					images.append(input_image)
					
					measurement = float(batch_sample[3])
					correction_factor = 0.2
					if(i == 1):
						measurement = measurement + correction_factor
					elif(i == 2):
						measurement = measurement - correction_factor
					measurements.append(measurement)
					
					# Augment data by flipping the image and negating the measurement
					images.append(cv2.flip(input_image,1))
					measurements.append(-1.0 * measurement)

			X_train = np.array(images)
			y_train = np.array(measurements)
			#import sklearn.utils.shuffle
			yield (X_train, y_train)
	
print('no. of training samples is  ', len(train_samples))
print('no. of validation samples is  ', len(validation_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("calling the train generator")
#print((next(train_generator)))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(image_shape)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Lenet Architecture
#model.add(Convolution2D(6,5,5, activation = "relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5, activation = "relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

# NVIDIA Architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = "relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

#model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
			
model.save('model.h5')

print('Time Taken to load the images and train the model : ', time.time() - start_time)