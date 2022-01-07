import re
from typing import Optional, List

import keras_segmentation
import tensorflow as tf
import cv2
import os
import numpy as np
import pathlib
import urllib
import zipfile

KERAS_APPLICATION_MODEL_NAMES = ['InceptionV3', 'VGG16', 'VGG19', 'ResNet50',
                                 'Xception', 'MobileNet', 'MobileNetV2', 'DenseNet121',
                                 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                                 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2',
                                 'ResNet152V2']
SEGMENTATION_MODEL_NAMES = list(keras_segmentation.models.model_from_name.keys())
LINEAR_MODEL_NAMES = ["linear" + str(i) for i in range(32)]
MODEL_NAMES = KERAS_APPLICATION_MODEL_NAMES + SEGMENTATION_MODEL_NAMES + ["test"] + LINEAR_MODEL_NAMES
CHAIN_GRAPH_MODELS = ["VGG16", "VGG19", "MobileNet"] + LINEAR_MODEL_NAMES
NUM_SEGMENTATION_CLASSES = 19  # Cityscapes has 19 evaluation classes

def data_reshape(src, target_shape):
    """ Scale images to target size, channel dimension not change

    Args:
        src (data arrays): an image array needs to be converted
        target_shape (tuple): output shape for each image
    """    
    return np.array([cv2.resize(src=image, dsize=target_shape, \
                    interpolation=cv2.INTER_LINEAR) for image in src])


def cifar10(batch_size, target_shape=None, num_imgs=None):
    """ Load cifar10 dataset with keras API.

    Args:
        batch_size (int): batch size of the keras data generator interface
        target_shape (None|(height, width)): if resize the image
        num_imgs (None|(train_num, test_num)): slice the number of images
    Returns:
        train data generator, validation data generator, the input shape and the num. of classes
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if num_imgs is not None:
        x_train = x_train[:num_imgs[0]]
        y_train = y_train[:num_imgs[0]]
        x_test = x_test[:num_imgs[1]]
        y_test = y_test[:num_imgs[1]]

    # Ref: https://keras.io/zh/utils/
    # default classes and shape
    num_classes = 10
    input_shape = (32, 32, 3) 
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    if target_shape is not None:
        input_shape = (target_shape[0], target_shape[1], input_shape[2]) # channel not change
        x_train = data_reshape(x_train, target_shape[::-1])
        x_test = data_reshape(x_test, target_shape[::-1])

    print(f"Shape of x_train: {x_train.shape}, shape of x_test: {x_test.shape}")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    train_generator = datagen.flow(
            x_train, y_train,
            batch_size=batch_size,
            shuffle=False)

    # Data generator for validation data
    validation_generator = datagen.flow(
            x_test, y_test,
            batch_size=batch_size,
            shuffle=False)
    return train_generator, validation_generator, input_shape, num_classes

def segmentation_dataset():
    """ Download and cache dataset for vgg_unet.
    """
    dataset_dir = os.path.abspath(__file__).split("/runtime/keras_model.py")[0]
    dataset_dir = os.path.join(dataset_dir, "dataset")

    if not pathlib.Path(dataset_dir).exists():
        pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        print("Create dataset dir {} for downloading segmentation dataset".format(dataset_dir))
        data_url = "https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip"
        try:
            urllib.request.urlretrieve(data_url, dataset_dir)
        except (urllib.error.URLError, IOError) as e:
            print("Dataset download error, you can manually download the zip from {}, and unzip it to {}" \
                .format(data_url, dataset_dir))
        
        with zipfile.ZipFile(os.path.join(dataset_dir, "dataset1.zip")) as zf:
            zf.extractall(dataset_dir)
        
    image_path = os.path.join(dataset_dir, "dataset1", "images_prepped_train")
    train_annotations = os.path.join(dataset_dir, "dataset1", "annotations_prepped_train")
    print("There are {} and {} images downloaded for train and test". \
                            format(len(os.listdir(image_path)), len(os.listdir(train_annotations))))
    return image_path, train_annotations


def linear_model(i):
    input = tf.keras.Input(shape=(224, 224, 3))
    x = input
    for i in range(i):
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=None, use_bias=False, name='conv' + str(i))(x)
    d = tf.keras.layers.GlobalAveragePooling2D(name='flatten')(x)
    predictions = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(d)
    return tf.keras.Model(inputs=input, outputs=predictions)


def get_keras_model(model_name: str, classes: int, input_shape: Optional[List[int]] = None, 
                                    include_top=True, weights=None):
    if model_name in KERAS_APPLICATION_MODEL_NAMES:
        model = eval("tf.keras.applications.{}".format(model_name))
        model = model(input_shape=input_shape, include_top=include_top, 
                                                weights=weights, classes=classes)
    elif model_name in SEGMENTATION_MODEL_NAMES:
        model = keras_segmentation.models.model_from_name[model_name]
        if input_shape is not None:
            # assert input_shape[2] == 3, "Can only segment 3-channel, channel-last images"
            model = model(n_classes=classes, input_height=input_shape[0], input_width=input_shape[1])
        else:
            model = model(n_classes=classes)
    else:
        raise NotImplementedError("Model {} not available".format(model_name))
    return model


def get_input_shape(model_name: str, batch_size: Optional[int] = None):
    model = get_keras_model(model_name, input_shape=None)
    shape = model.layers[0].input_shape
    if batch_size is not None:
        shape[0] = batch_size
    return shape