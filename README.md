# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioural-cloner.ipynb  -  containing the script to create and train the model
* drive.py   -  for driving the car in autonomous mode
* model_tune_101.h5   -  containing a trained convolution neural network
* model_tune_101.mp4 - the output video
* README.md   -  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_tune_101.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the NVIDIA architecture(Defined in the last cell of the notebook). Images are preprocessed by a Lambda layer to normalize the data and the top and bottom pixels are cropped to avoid unnecessary data like sky, trees and car dashboard.

The model includes RELU layers to introduce nonlinearity. 

#### 2. Attempts to reduce overfitting in the model

Although the model has 4 epochs configured, the output of every epoch is stored and based on the loss details in training and validation, the best fit result is chosen.  I got this idea from https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project and it proved to be very useful where I had the option to test the simlator with the output of different epochs and observe the result.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (defined in the last cell of the notebook).

#### 4. Appropriate training data

I've tried the following training data:

* Track1_forward_2_laps
* Track1_reverse_2_laps
* Track1_shoulder_to_road_1_lap
* Track2_forward_2_laps
* Track1_smooth_curves
* Track1_smooth_curves_2
* Track1_smooth_curves_3

When testing the simulator the car was crossing the road track during sharp turns and other turns. To avoid that, I've included 3 iterations of smooth curve data. Some of them include the areas where the car went out of the track. 

The lap was a loop with lot of left turns. The reverse laps will ensure that the model learns how to turn right as well. The model should also learn to come back to the center of the road if it reaches the corner of the road. To train that I have included a data set where I drove the car from the side of the road to the center of the road.

Track 2 has lot of curves and slopes and was more challenging that Track 1. I've recorded 2 laps of this track so that the model doesn't overfit to Track 1.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I implemented the NVIDIA architecture. Images were preprocessed using a Lamda layer to normalize and cropped the top and bottom portions of the image to remove unwanted content.

NVIDIA architecture has 5 Convolution layers where each of the layer is followed by a RELU activation layer. This is followed by a Flatten layer and 4 Dense layers. The 4th layer is to take a single output and was introduced for the purposes of this project.

I've used Adam optimizer with Mean Squared Error as the loss function.

The car was able to run autonomously in most part of the road with the initial set of data points(excluding smooth curves data) but it was going out of the road on curves.

Data was visualized(method name is visualize_steering_angles_histogram())  and found that most part of the data was having steering angle as 0. So the model learnt a lot about going straight and not much about turnings.

Implemented a method(method name - flatten_data_point() ) to flatten some of the data points. After lot of trials I found that flattening the bin with steering angle values of 0 to 0.1 and excluding the udacity data resulted in better result when tested on the simulator and also in the histogram.

I've also used all the 3 camera images. The left and right camera images are used with a correction factor of 0.2. Each of the images are flipped using Open CV and the steering angle measurement was negated to get more image data.

I also attempted to change the brightness of the image by following https://github.com/budmitr/CarND-Behavioral-Cloning but the results were not too good. So I decided not to use that.

A generator is introduced to ensure that only few images are stored in memory. This also showed a performance improvement and I was able to train the model in my laptop few times.

#### 2. Final Model Architecture
Defined above.

#### 3. Creation of the Training Set & Training Process

Defined the training data points in the above section "4. Appropriate training data".

Training and test data are split using "train_test_split" form sklearn toolkit. Test set contains 20% of the data.

All these images were shuffled to ensure that the time series of images doesn't introduce any bias to the network.
