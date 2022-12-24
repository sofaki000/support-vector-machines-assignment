import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

from utilities.augmentation_utilities import get_symmetry_horizontal


def plot_normal_pics(X_train):
    datagen = ImageDataGenerator()
    datagen.fit(X_train)
    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(X_train[i].T)
    pyplot.savefig("og_images.png")
    pyplot.clf()

def plot_normalized_pics(X_train):
    datagen = ImageDataGenerator(samplewise_std_normalization=True)
    datagen.fit(X_train)
    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(X_train[i].T)
    pyplot.savefig("normalized_images.png")
    pyplot.clf()

def plot_rotated(X_train):
    datagen = ImageDataGenerator(rotation_range=359)
    datagen.fit(X_train)
    for i in range(len(X_train)):
        X_train[i] = X_train[i].reshape(1,3,32,32)

    aug_iter = datagen.flow(X_train, batch_size=1)

    for i in range(0, 4):
        image = next(aug_iter)[0].astype('uint8')
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(np.squeeze(image).T)
    pyplot.savefig("rotated_images.png")

    pyplot.clf()

def plot_vertically_scrolled(X_train):
    datagen = ImageDataGenerator(height_shift_range=0.5)
    datagen.fit(X_train)
    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(X_train[i].T)
    pyplot.savefig("vertical_scrolled.png")
    pyplot.clf()

def print_data_augmentation_images(X_train, y_train):
    X_clean = X_train
    X_train = X_clean

    for i in range(0, 4):
        pyplot.subplot(220 + 1 + i)
        pyplot.imshow(X_train[i].T)
    pyplot.savefig("normal_photos.png")
    pyplot.clf()

    ######### symmetrical horizontal
    augmented_data = get_symmetry_horizontal(X_train)

    # X_train = X_clean
    # datagen = ImageDataGenerator(featurewise_center=True)
    # datagen.fit(X_train)
    # for i in range(0, 4):
    #     pyplot.subplot(220 + 1 + i)
    #     pyplot.imshow(X_train[i].T)
    # pyplot.savefig("featurewise_center.png")
    # pyplot.clf()
    # ############
    #
    # X_train = X_clean
    # datagen = ImageDataGenerator(samplewise_center=True)
    # datagen.fit(X_train)
    # pyplot.clf()
    # for i in range(0, 4):
    #     pyplot.subplot(220 + 1 + i)
    #     pyplot.imshow(X_train[i].T)
    # pyplot.savefig("samplewise_center.png")

    print("DONE")

def normalize_images(X_train):
    datagen = ImageDataGenerator(samplewise_std_normalization=True)
    datagen.fit(X_train)
    return X_train

def rotate_images(X_train   ):
    datagen = ImageDataGenerator(rotation_range=359)
    datagen.fit(X_train)
    return X_train


def vertical_scroll_images(X_train):
    datagen = ImageDataGenerator(height_shift_range=0.5)
    datagen.fit(X_train)

    return X_train

def symmetry_horizontal(X_train):
    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen.fit(X_train)

    return X_train


def featurewise_center(X_train):
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.fit(X_train)
    return X_train

def samplewise_center(X_train):
    datagen = ImageDataGenerator(samplewise_center=True)
    datagen.fit(X_train)
    return X_train


