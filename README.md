# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Behavioral Cloning

### Overview

In this project, we have to train a car using the machine learning techniques and test the same if it is able to drive on the track using Udacity Simulator

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/)
- [keras] (https://keras.io)


### Dataset

For the initial few runs, I have used Udacity Data and the model was not performing upto the mark. Later on, Annie Flipp(classmate) has provided me the recovery data she collected. Training the model using both the Udacity Data and Annie's Data has improved the model performance a lot and the model works on both the tracks 1 and 2.

Here is the screen-shot of the data set used:

![alt tag] (https://github.com/vamckorivi/CarND-BehavioralCloning/blob/master/images/car_images.png)

###Preprocessing

To understand the data, we have plotted the histogram of the angles and clearly indicates that the data with 0 angles is huge and it has a left turn bias. There are 3 camera angles provided to us - Center, Left, Right and the corresponding images and the angle corresponding to the image.

Below is the plot of the steering angles and it clearly shows the training data has relatively high zero angles which is not a balanced training set.

![alt tag] (https://github.com/vamckorivi/CarND-BehavioralCloning/blob/master/images/P3B_hist.png)

While testing on the track, our prediction was not so accurate in the initial iterations and the car was going out of the track very easily. For the initial iteration, we have tested with only Udacity Data and the results were really poor.

To improve the model, we had to have the more training data or use the existing data and augment(Thanks to Vivek who has a written an awesome blog post about image augmentation). I have directly jumped on to the image augmentation to see how it works. In the next couple of iterations, I have used only flipping the images and then randomly choosing the left or right images and adding the angle deviations. Till this iteration, we were using basic keras_lab model and without the fit_generator, all the data was produced during preprocessing step.

After couple of iterations and the pain of the model not working, we have shifted to the most popular model in the slack channel - NVIDIA model. 


I have tried to keep the augmentations as much low as possible and have kept it to only two things
1. Randomly flipping images
2. Adding angles when choosing left(+0.25)/right(-0.25) images
3. Resized the images to 64,64. This was done only for the purpose of speeding up the training.

As part of the preprocessing step, I have followed Annie's method of removing all the zero angles. For the near zero angles like between -.15 to .15 degree, added around 10-20 images with small deviations to the angle and adding that to the data set. This preprocessing is done for both the Udacity Data and Recovery Data and this upsampling of the data has actually improved the model a lot.


### Training

Keras lab model was used for the initial runs. After lot of trial and error with that model, switched to the most popular model NVIDIA. Below is the architecture of the NVIDIA model used. 

![alt tag] (https://github.com/vamckorivi/CarND-BehavioralCloning/blob/master/images/nvidia_architecture.png)

Lambda Normalization which helped to increase the speed of training.


### Hyper Parameters
Number of epochs used in the final model is 20. Tried various epochs like 3,5,9,10,15,20 and finally sticked to 20 as it gave the best result. Though the model was waivering sometimes when number of epochs increased but the final model with 20 epochs gave the desired results.
Adam Optimizer is used. Kept the default learning rate for most of the models/iterations of training but did try 0.001 and 0.0001. Default learning rate of 0.01 was giving good results, so we used the same in the final model.
Batch Size of 256 is used in the final model. Different batch sizes were used when trained on the macbook pro like 32,64. But when we started using AWS, Batch size of 128 and 256 were used


### Tracks Run
Model was run both Track 1 and Track 2 successfully. Though, there was no brightness augmentation was not applied during training, model performed very well on Track 2 which was acutally quite suprising!

Track Run Video:

[![IMAGE ALT TEXT HERE](https://i.ytimg.com/vi/l6P1NvL_8kY/hqdefault.jpg?custom=true&w=196&h=110&stc=true&jpg444=true&jpgq=90&sp=68&sigh=7YqVN_j4ISwBDBdJ9jA3i5U9LCQ)](https://youtu.be/l6P1NvL_8kY)

### Take away from the project

1. The importance of training data. In P2, the training data was reasonably enough and applying little augmentations like the tilting images to get different angles was sufficient. In this project, we had to eliminate the zero angle which was causing overfitting, remove the left bias. This was a huge learning curve on P3 regarding how really important is training data. I tried for a week to figure out with just Udacity data but it was not enough for the model to work. Recovery data was so useful along with the little augmentations made.
2. Resizing of the images has saved lot of time when training. When NVIDIA size of 66,208 was used, each epoch was taking around 200 seconds on AWS, but when switched to 64,64, it took on an average 60 seconds per each epoch.
3. I had to stick with NVIDA model after couple of days tweaking my own model to make the submission on time. Played around with comma.ai model too but for P3, we used Nvidia, but in real time we had to figure out our own training model. In future, I would like to come up with my own tiny model which could be good enough.
4. I had to spend nearly a week to figure out why the model was predicting constant angle. I reached out to classmates on slack and my mentor for resolution. With the comments received from them, I tried various things of upsampling, downsampling, trying different models. But finally, I figured out it was due to the normalization errors I made. It was a very small thing but took a lot of time to figure out. Debugging in Machine Learing is really hard!
5. Running with lowest screen resolution(640*480) and fastest Graphics Quality helped to perform the model better.
 



