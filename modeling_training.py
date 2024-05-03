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

#Compiling the model with specified parameters.
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.summary()

#Setting up ratio for train/test split.
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

#Calculating number of samples in each set.
num_samples = len(preprocessed_images_all)
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = int(test_ratio * num_samples)

#Shuffling the dataset, for randomness.
indices = np.random.permutation(num_samples)

#Splitting the dataset into indices.
train_indices = indices[:num_train_samples]
val_indices = indices[num_train_samples:num_train_samples+num_val_samples]
test_indices = indices[-num_test_samples:]

#Train/Test split.
x_train, y_train = preprocessed_images_all[train_indices], preprocessed_labels[train_indices]
x_val, y_val = preprocessed_images_all[val_indices], preprocessed_labels[val_indices]
x_test, y_test = preprocessed_images_all[test_indices], preprocessed_labels[test_indices]

print('Training set shape:', x_train.shape, y_train.shape)
print('Validation set shape:', x_val.shape, y_val.shape)
print('Test set shape:', x_test.shape, y_test.shape)

#Training the model
hist = model.fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 32,
    validation_data = (x_val, y_val)
)

#Visualizing the training hitory.
plt.figure(figsize=(12, 6))

#Plotting training and validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(hist.hist['accuracy'])
plt.plot(hist.hist['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid()