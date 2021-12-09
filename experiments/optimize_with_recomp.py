from models.keras_model import get_keras_model
import tensorflow as tf
import numpy as np
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
