import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# disable the log of tf c++ core
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import tensorflow as tf
import numpy as np
# turn on memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras import backend as K
from keras_model import cifar10, get_keras_model

# GPU memory limitation
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.676))) 
K.set_session(sess)

# Configure the memory optimizer and dependency optimizers
# config = tf.ConfigProto()
# config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
# config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
# K.set_session(tf.Session(config=config))

def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="VGG16", help="VGG16, vgg_unet and MobileNet")
    parser.add_argument('--device-id', type=str, default='0', help="GPU device index")
    parser.add_argument('--device-memory', type=int, default=10240,
                        help="Device memory limitation in MB")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help="If set, print STR log")
    _args = parser.parse_args()
    
    return _args

if __name__ == "__main__":
    args = extract_params()
    print("Parsing arguments: {}".format(args._get_kwargs()))
    
    # only support single GPU currently
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # prepare dataset
    train_generator, validation_generator, input_shape, classes = cifar10(args.batch_size, (224, 224))
    model = get_keras_model(model_name=args.model_name, input_shape=input_shape, 
                                classes=classes, include_top=True, weights=None)

    # training
    model.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
    history = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=validation_generator,
            verbose=1)