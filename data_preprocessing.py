#importing packages
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import random

#Function to load batch files
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

#Image preprocessor
def preprocess_images(images):
    # Resizing to 120 x 120
    resized_images = np.array([cv2.resize(img.reshape(3, 32, 32).transpose(1, 2, 0), (120, 120)) for img in images])

    # Normalize pixel values to the range
    normalized_images = resized_images / 255.0

    return normalized_images

#Load and preprocess the test batch
test_batch_file_path = 'cifar-10-batches-py/test_batch'
test_batch = load_batch(test_batch_file_path)
test_images = test_batch[b'data']
test_labels = np.array(test_batch[b'labels'])
preprocessed_test_images = preprocess_images(test_images)

#Load CIFAR-10 training batches
preprocessed_batches = []
for i in range(1,6):
    #Loading each batch
    batch_file_path = f'cifar-10-batches-py/data_batch_{i}'
    batch = load_batch(batch_file_path)
    images = batch[b'data']

    #Preprocessing
    preprocessed_images = preprocess_images(images)

    #Adding preprocessed batch to list
    preprocessed_batches.append(preprocessed_images)

#Concatenate batches along batch axis
preprocessed_images_all = np.concatenate(preprocessed_batches, axis=0)

#Preprocess and concatenate labels
preprocessed_labels = np.concatenate([np.array(batch[b'labels'])for batch in [load_batch(f'cifar-10-batches-py/data_batch_{i}')for i in range(1,6)]])

#Getting class counts
class_counts = Counter(preprocessed_labels)

#Extracting labels and their counts from class_counts dictionary. Seperating into two seperate lists.
labels, counts = zip(*class_counts.items())

#Plotting class distribution
plt.figure(figsize=(10, 6))
plt.bar(labels, counts)
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.title("Class Distribution")
plt.xticks(range(10)) #10 classes in total in CIFAR-10
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Randomly selecting samples to visualize and inspect
num_samples_per_class = 5
class_sample_dictionary = {}
for label in range(10):
    indices = np.where(preprocessed_labels == label)[0]
    # Limit the number of samples per class to 5
    samples = random.sample(list(indices), min(len(indices), num_samples_per_class))
    class_sample_dictionary[label] = samples

# Plot sample images
plt.figure(figsize=(15,10))
total_samples = num_samples_per_class * len(class_sample_dictionary)
for i, (label, indices) in enumerate(class_sample_dictionary.items(), start=1):
    for j, index in enumerate(indices, start=1):
        subplot_index = (i - 1) * num_samples_per_class + j
        plt.subplot(len(class_sample_dictionary), num_samples_per_class, subplot_index)
        plt.imshow(preprocessed_images_all[index])
        plt.title(f"Class {label}")
        plt.axis('off')
plt.tight_layout()
plt.show()


#Iterate through and print class counts
for label, count in class_counts.items():
    print(f"Class {label}: {count} images")

#View the shape of the images and labels arrays
print('Preprocessed images shape: ', preprocessed_images_all.shape)
print('Preprocessed labels shape: ', preprocessed_labels.shape)