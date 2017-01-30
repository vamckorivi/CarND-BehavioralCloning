#all the import statements here
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle


#Read driving_log.csv and the images(Udacity Training Set, No New images generated for this assignment)

data = pd.read_csv("./data/driving_log.csv")
#recovery data provided by Annie Flippo
data_recovery = pd.read_csv("./IMG_recovery/driving_log_recovery.csv")


#model goes here.
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Lambda, ELU
def keras_lab():
    model = Sequential()
    model.add(Convolution2D(32,3,3,border_mode='valid', input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(43))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    model.compile('adam','mse')
    return model
    
def nvidia():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 -1,input_shape = (64,64,3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2),  border_mode="valid", init='he_normal'))
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(Flatten())
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
#     model.compile('adam', 'mse')
    
    return model
    
#the functions here
def preprocessed_data(row):
    steering = row['steering']
    #print("preprocess_daa")
    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    #print(camera)
    image_loc = row[camera][0]
    #print(image_loc)
    image_loc = image_loc.strip()
    #print(image_loc)
    image = cv2.imread(image_loc)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    #image = np.array(image)
    #print("hi1")
    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Crop, resize and normalize the image
    #image = image[25:140, :, :]
    image = image[55:135, :, :]
    #image = cv2.resize(image,(208,66))
    image = cv2.resize(image,(64,64))

    #image  = image/255.-.5
    #image  = image/127.5-1
    return image, steering


def preprocessed_valid_data(row):
    steering = row['steering']
    image_loc = row['center'][0]
    #print(image_loc)
    image_loc = image_loc.strip()
    #print(image_loc)
    image = cv2.imread(image_loc)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Crop, resize and normalize the image
    #image = image[25:140, :, :]
    image = image[55:135, :, :]
    #image = cv2.resize(image,(208,66))
    image = cv2.resize(image,(64,64))
    #image  = image/127.5-1
    return image, steering

def train_generaotr(data_df):
    #batch_images = np.zeros((batch_size, 66, 208, 3))
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_angles = np.zeros(batch_size)
    #print("in train")
    while True:
        for i in range (batch_size):
            # Randomly get a sample from the input data
            #print(i)
            idx = np.random.randint(len(data_df))

            # reset_index sets this data_df starting row to 0
            data_row = data_df.iloc[[idx]].reset_index()
            img1, angle1 = preprocessed_data(data_row)

            batch_images[i] = img1
            batch_angles[i] = angle1
            
        yield batch_images, batch_angles
        
def valid_genertor(data_df):
    #batch_images = np.zeros((batch_size, 66, 208, 3))
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_angles = np.zeros(batch_size)
    #print("in train")
    while True:
        for i in range (batch_size):
            # Randomly get a sample from the input data
            #print(i)
            idx = np.random.randint(len(data_df))

            # reset_index sets this data_df starting row to 0
            data_row = data_df.iloc[[idx]].reset_index()
            img1, angle1 = preprocessed_valid_data(data_row)

            batch_images[i] = img1
            batch_angles[i] = angle1
            
        yield batch_images, batch_angles


data = data.sample(frac=1).reset_index(drop=True)
#This piece of code is implemented with the help of annie flippo's blog
#Also, training data is taken from annie flippo rather than generating it again
data_right = []
data_left = []
data_center = []

#iterating through the data
for i in range(len(data)):
    center_img = data["center"][i]
    left_img = data["left"][i]
    right_img = data["right"][i]
    steering_angle = data["steering"][i]
    
    #tried different steering angle combinations like 0.15, 0.2,0.1 but 0.15 steering angle 
    #gave better results and went with it
    if (steering_angle > 0.15):
        data_right.append([center_img, left_img, right_img, steering_angle])
        for i in range(10):
            new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            data_right.append([center_img, left_img, right_img, new_angle])

    elif (steering_angle < -0.15):
        data_left.append([center_img, left_img, right_img, steering_angle])
        for i in range(20):
            new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            data_left.append([center_img, left_img, right_img, new_angle])

    else:
        if (steering_angle != 0):
            for i in range(5):
                new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                data_center.append([center_img, left_img, right_img, new_angle])

#iterating through the recovery data provided by annie flippo
for i in range(len(data_recovery)):
    center_img = data_recovery["center"][i]
    left_img = data_recovery["left"][i]
    right_img = data_recovery["right"][i]
    steering_angle = data_recovery["steering"][i]
    
    #tried different steering angle combinations like 0.15, 0.2,0.1 but 0.15 steering angle 
    #gave better results and went with it
    if (steering_angle > 0.15):
        data_right.append([center_img, left_img, right_img, steering_angle])
        for i in range(10):
            new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            data_right.append([center_img, left_img, right_img, new_angle])

    elif (steering_angle < -0.15):
        data_left.append([center_img, left_img, right_img, steering_angle])
        for i in range(20):
            new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            data_left.append([center_img, left_img, right_img, new_angle])

    else:
        if (steering_angle != 0):
            for i in range(5):
                new_angle = steering_angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                data_center.append([center_img, left_img, right_img, new_angle])

from sklearn.model_selection import train_test_split


data_center = pd.DataFrame(data_center, columns=["center", "left", "right", "steering"])
data_left = pd.DataFrame(data_left, columns=["center", "left", "right", "steering"])
data_right = pd.DataFrame(data_right, columns=["center", "left", "right", "steering"])
data = [data_center, data_left, data_right]
data = pd.concat(data, ignore_index=True)


                                   

batch_size=256

# data = data.sample(frac=1).reset_index(drop=True)
#splitting data into 80%training, 20%validation
# training_data_index = int(data.shape[0]*0.8)
# training_data = data.loc[0:training_data_index-1]
# validation_data = data.loc[training_data_index:]

#splitting data into 80%training, 20%validation
training_data, validation_data = train_test_split(data, test_size=0.2)

training_data  = pd.DataFrame(training_data,columns=["center", "left", "right", "steering"])
validation_data  = pd.DataFrame(validation_data,columns=["center", "left", "right", "steering"])

val_size = len(validation_data)
#print(training_data.shape)
#print(validation_data.shape)

#testing generator with yield
#gener_exam = get_primes(5)
#print(gener_exam)
training_generaotr = train_generaotr(training_data)
#print(train_generaotr)
validation_generator = valid_genertor(validation_data)
#print(validation_generator)
model  = nvidia()
#model  = get_model()
#model  = keras_lab()

history = model.fit_generator(training_generaotr, validation_data=validation_generator,
                              samples_per_epoch=20224, nb_epoch=20, nb_val_samples=3000)
                            

from keras.models import model_from_json
import json
json_string = model.to_json()
with open("model.json", 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights("model.h5")