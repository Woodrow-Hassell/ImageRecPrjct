import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import random
import tensorflow as tf

from data_preprocessing import load_batch, preprocess_images, preprocessed_images_all, preprocessed_labels

#Load in the model.
resnet_mdl = tf.keras.applications.ResNet50(
    include_top = False,
    weights = 'imagenet', 
    input_shape = (120, 120, 3)
)

#Freezing pretrained weights within each layer so pretrained knowledge is retained.
for layer in resnet_mdl.layers:
    layer.trainable = False

#Creating custom fully connected layers.
x = tf.keras.layers.Flatten()(resnet_mdl.output) #Reshapes output of ResNet50 base into 1d vector.
x = tf.keras.layers.Dense(256, activation = 'relu')(x) #Serves as a hidden layer to capture patterns.
predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x) #Generates probability distribution over the classes. 

#Creating the final model, combining the ResNet-50 base with the custom layers.
model = tf.keras.Model(inputs=resnet_mdl.input, outputs=predictions)

#Compiling the model with specified parameters
model.compile(optimizer = 'adam', loss = 'spars_categorical_crossentropy', metrics = ['accuracy'])

model.summary()