import argparse
import os
from requests import options
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from keras_model import cifar10, get_keras_model, segmentation_dataset
from str_config import ConfigHandler
from transformer_model import build_model, load_dataset

# GPU memory limitation
# sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.676))) 
# K.set_session(sess)

# Configure the memory optimizer and dependency optimizers
# config = tf.ConfigProto()
# config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
# config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
# K.set_session(tf.Session(config=config))

# Tensorboard for profiling
# import tensorboard
# tensorboard.__version__
# log_dir = "logs/fit-resnet101"# + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                                         histogram_freq=1,
#                                                         profile_batch=10)
from tensorflow.python.client import timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="VGG16", help="VGG16, vgg_unet and MobileNet")
    # parser.add_argument('--device-id', type=str, default='0', help="GPU device index")
    # parser.add_argument('--device-memory', type=int, default=10240,
    #                     help="Device memory limitation in MB")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help="If set, print STR log")
    parser.add_argument("--strategy", type=str, default="str", help="dynprog|str|str-app|checkmate|capuchin|chen-heurist|none")
    parser.add_argument("--profile", type=bool, default=False, help="set True if only want to print model layer names")
    _args = parser.parse_args()
    
    return _args

if __name__ == "__main__":
    args = extract_params()
    print("Parsing arguments: {}".format(args._get_kwargs()))
    
    conf_handle = ConfigHandler(args.model_name, args.batch_size)
    conf_handle.set_strategy(args.strategy)

    # prepare dataset
    if args.model_name == "vgg_unet":
      # Load segmentation dataset
      image_path, annotation_path = segmentation_dataset()
      model = get_keras_model(model_name=args.model_name, input_shape=(320, 640), 
                                  classes=50, include_top=True, weights=None)

      model.train(train_images=image_path,
              train_annotations=annotation_path,
              batch_size=args.batch_size,
              steps_per_epoch=64,
              epochs=1)
    elif args.model_name == "transformer":
      # using imdb dataset to train transformer
      vocab_size=512
      max_len=128
      train, test = load_dataset(vocab_size, max_len)
      x_train, x_train_masks, y_train = train

      model = build_model(vocab_size, max_len)
      if args.profile:
        model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'],
                      options=run_options, # for write out timeline.json
                      run_metadata=run_metadata)
      else:
        model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
      es = tf.keras.callbacks.EarlyStopping(patience=3)
      model.fit([x_train, x_train_masks], y_train,
                batch_size=args.batch_size, epochs=args.epochs, 
                steps_per_epoch=np.math.ceil(len(x_train) / args.batch_size), callbacks=[es])

    else:
      train_generator, validation_generator, input_shape, classes = cifar10(args.batch_size, (224, 224), [40000, 10000])
      model = get_keras_model(model_name=args.model_name, input_shape=input_shape, 
                                  classes=classes, include_top=True, weights=None) 
      # training
      if args.profile:
        model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(),
                metrics=['accuracy'], 
                options=run_options, # for write out timeline.json
                run_metadata=run_metadata)
      else:
        model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
      history = model.fit(
              train_generator,
              epochs=args.epochs,
              validation_data=validation_generator,
              verbose=1)
      
    if args.profile:
      # Write out names of each layer
      # Tips: for transformer, this layer is too coarse-grained. We will parse second-level names with tool.py
      with open(f"layer_names_{args.model_name}", "w") as f:
          for layer in model.layers:
            f.write(layer.name + "\n")
      # Save timeline file
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open(f"timeline-{args.model_name}-{args.batch_size}.json", "w") as f:
        f.write(trace.generate_chrome_trace_format())