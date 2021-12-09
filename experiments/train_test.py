import os,sys
# os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='1' 
# if 'absl.logging' in sys.modules:
#   import absl.logging
#   absl.logging.set_verbosity('debug')
#   absl.logging.set_stderrthreshold('debug')

import tensorflow as tf
import numpy as np

from tensorflow.python.client import timeline
import datetime

# import tensorboard
# tensorboard.__version__

from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.INFO)

batch_size = 32
image_size = 32
epochs = 1
num_classes = 10

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

# using gradient checkpoint
# memory_saving_gradients.MIN_CHECKPOINT_NODE_SIZE = 100
# K.gradients = memory_saving_gradients.gradients_memory
# K.__dict__['gradients'] = memory_saving_gradients.gradients_memory
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

datagen = tf.keras.preprocessing.image.ImageDataGenerator()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Ref: https://keras.io/zh/utils/
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"len of x_train: {x_train.shape[0]}, len of x_test: {x_test.shape[0]}")
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
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

model = tf.keras.applications.VGG16(
    include_top=True,
    weights=None,
    input_shape=((image_size, image_size, 3)),
    classes=10)

# convert to graph mode
# graph_model = tf.function(model)
# model.summary()

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# 需要传入run options才能返回得到run metadata
run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()

model.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
            # options=run_options,
            # run_metadata=run_metadata)
history = model.fit(
      train_generator,
      epochs=epochs,
      validation_data=validation_generator,
      verbose=0)
# tf.io.write_graph(tf.get_default_graph().as_graph_def(), "model_dir", "vgg.pbtxt")
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open("timeline.json", 'w') as f:
#     f.write(ctf)
"""
image_size = 224
model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=((image_size,image_size,3))
)
for layer in model.layers:
    print(layer, layer.trainable)

# Load the normalized images
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 50
val_batchsize = 10

dataset_dir = "/data1/zongzan/dataset_v/"
# Data generator for training data
train_generator = train_datagen.flow_from_directory(
        dataset_dir + "train",
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=False)

# Data generator for validation data
validation_generator = validation_datagen.flow_from_directory(
        dataset_dir + "val",
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
# model.fit(train_generator, validation_generator, epochs=10)
# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=
         train_generator.samples/train_generator.batch_size,
      epochs=20,
      validation_data=validation_generator, 
      validation_steps=
         validation_generator.samples/validation_generator.batch_size,
      verbose=1)
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
"""