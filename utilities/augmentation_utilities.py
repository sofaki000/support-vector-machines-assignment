import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def plot_augmented_data(fileName , augmented_data):
    for i in range(len(augmented_data)):
        if type(augmented_data[0]) == np.ndarray is True and augmented_data[i].shape==(3,32,32):
            continue

        augmented_data[i] = augmented_data[i][0].reshape(3, 32, 32)

    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(augmented_data[i].T)
    pyplot.savefig(f'{fileName}.png')
    pyplot.clf()

def plot_normal_data(fileName , augmented_data):
    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(augmented_data[i].T)
    pyplot.savefig(f'{fileName}.png')
    pyplot.clf()



def get_augmented_data_from_imageDataGenerator(datagen, X_train):
    datagen.fit(X_train)
    augmented_data = []
    number_of_items = len(X_train)
    counter = 0
    for X_batch in datagen.flow(X_train, batch_size=1):
        augmented_data.append(X_batch)
        counter += 1
        if counter == number_of_items:
            break

    return augmented_data

def get_symmetry_horizontal(X_train):
    datagen = ImageDataGenerator(horizontal_flip=True)
    return get_augmented_data_from_imageDataGenerator(datagen, X_train)

def get_normalized_images(X_train):
     datagen = ImageDataGenerator(samplewise_std_normalization=True)
     return get_augmented_data_from_imageDataGenerator(datagen, X_train)

def get_rotated_images(X_train):
    datagen = ImageDataGenerator(rotation_range=359)
    return get_augmented_data_from_imageDataGenerator(datagen, X_train)

def get_featurewise_center(X_train):
    datagen = ImageDataGenerator(featurewise_center=True)
    return get_augmented_data_from_imageDataGenerator(datagen, X_train)