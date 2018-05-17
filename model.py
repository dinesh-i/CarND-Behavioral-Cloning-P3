import csv
import cv2
import numpy as np
import time

file_contains_header = True

dev_mode = True
no_of_images_to_read_in_dev_mode = 1000

# Read the data collected in csv file

lines = []

with open('../CarND-Behavioral-Cloning-P3-data/data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	for line in reader:
		lines.append(line)
	
# Update the image path from absolute path to relative path as we plan to train the model in AWS and the absolute path will be different there

images = []
measurements = []

start_time = time.time()

relative_path = '../CarND-Behavioral-Cloning-P3-data/data/IMG'
for index, line in enumerate(lines):
	if(index == 0 and file_contains_header):
		continue
		
	if(dev_mode and index > no_of_images_to_read_in_dev_mode):
		break
	
	for i in range(3):
		image_file_name = line[i].split('/')[-1]
		relative_file_name = relative_path + '/' + image_file_name
		input_image = cv2.imread(relative_file_name)
		images.append(input_image)
		
		image_shape = input_image.shape
		
		measurement = float(line[3])
		correction_factor = 0.2
		if(i == 1):
			measurement = measurement + correction_factor
		elif(i == 2):
			measurement = measurement - correction_factor
		measurements.append(measurement)
	
print('no. of images ', len(images))
print('no. of measurements ', len(measurements))
print('image shape is ', image_shape)

print('Time Taken to read the images and measurements : ', time.time() - start_time)

# Augment the images and measurements by flipping the image and negating the steering measurement
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(-1.0 * measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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

model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=5)

model.save('model.h5')

