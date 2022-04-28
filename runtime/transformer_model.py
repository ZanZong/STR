#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback

# TensorFlow 1.x doesn't support the official implementation of Transformer,
# here we choose to use the Keras implementation from 
#  https://raw.githubusercontent.com/LongmaoTeamTf/ \
#  deep_recommenders/master/deep_recommenders/keras/models/nlp/transformer.py

######################
# Multi-head attention
######################
class Embedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, model_dim, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        return embeddings

    def compute_output_shape(self, input_shape):

        return input_shape + (self._model_dim,)


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2**32+1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs
    
    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs, **kwargs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks) # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul) # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)
        
        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape



class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
        
        queries_linear = K.dot(queries, self._weights_queries) 
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)
        
        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
            
        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape



######################
# Transformer
######################

class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class PositionWiseFeedForward(Layer):
    
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bias_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_inner")
        self.bias_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim


class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape



class Transformer(Layer):

    def __init__(self,
                 vocab_size,
                 model_dim,
                 n_heads=8,
                 encoder_stack=6,
                 decoder_stack=6,
                 feed_forward_size=2048,
                 dropout_rate=0.1,
                 **kwargs):

        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        self.EncoderPositionEncoding = PositionEncoding(self._model_dim)
        self.EncoderMultiHeadAttentions = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms0 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        self.EncoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms1 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        self.DecoderPositionEncoding = PositionEncoding(self._model_dim)
        self.DecoderMultiHeadAttentions0 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms0 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderMultiHeadAttentions1 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms1 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms2 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        super(Transformer, self).build(input_shape)
        
    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = self.EncoderPositionEncoding(embeddings)
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = self.EncoderMultiHeadAttentions[i]
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = self.EncoderLayerNorms0[i](attention_out)
            # Feed-Forward
            ff = self.EncoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.EncoderLayerNorms1[i](ff_out)

        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = self.DecoderPositionEncoding(embeddings)
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = self.DecoderMultiHeadAttentions0[i]
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = self.DecoderLayerNorms0[i](masked_attention_out)

            # Multi-head-Attention
            attention = self.DecoderMultiHeadAttentions1[i]
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = self.DecoderLayerNorms1[i](attention_out)

            # Feed-Forward
            ff = self.DecoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.DecoderLayerNorms2[i](ff_out)

        # Pre-SoftMax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, encoder_inputs, decoder_inputs, **kwargs):
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self._vocab_size

    def get_config(self):
        config = {
            "vocab_size": self._vocab_size,
            "model_dim": self._model_dim,
            "n_heads": self._n_heads,
            "encoder_stack": self._encoder_stack,
            "decoder_stack": self._decoder_stack,
            "feed_forward_size": self._feed_forward_size,
            "dropout_rate": self._dropout_rate
        }
        base_config = super(Transformer, self).get_config()
        return {**base_config, **config}


class Noam(Callback):

    def __init__(self, model_dim, step_num=0, warmup_steps=4000, verbose=False):
        self._model_dim = model_dim
        self._step_num = step_num
        self._warmup_steps = warmup_steps
        self.verbose = verbose
        super(Noam, self).__init__()

    def on_train_begin(self, logs=None):
        logs = logs or {}
        init_lr = self._model_dim ** -.5 * self._warmup_steps ** -1.5
        K.set_value(self.model.optimizer.lr, init_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self._step_num += 1
        lrate = self._model_dim ** -.5 * K.minimum(self._step_num ** -.5, self._step_num * self._warmup_steps ** -1.5)
        K.set_value(self.model.optimizer.lr, lrate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            lrate = K.get_value(self.model.optimizer.lr)
            print(f"epoch {epoch} lr: {lrate}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
    

def label_smoothing(inputs, epsilon=0.1):
    """目标平滑"""
    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


##################
#### train on IMDB
##################
class TextGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, mask, y, batch_size):
        self.x = x
        self.y = y
        self.mask = mask
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        mask_x = self.mask[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return (batch_x, mask_x), batch_y

    def on_epoch_end(self):
        pass

def load_dataset(vocab_size, max_len):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = tf.keras.utils.to_categorical(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, x_train_masks, y_train), (x_test, x_test_masks, y_test)


def build_model(vocab_size, max_len, model_dim=8, n_heads=4, encoder_stack=3, decoder_stack=3, ff_size=50):
    encoder_inputs = tf.keras.Input(shape=(max_len,), name='encoder_inputs')
    decoder_inputs = tf.keras.Input(shape=(max_len,), name='decoder_inputs')
    outputs = Transformer(
        vocab_size,
        model_dim,
        n_heads=n_heads,
        encoder_stack=encoder_stack,
        decoder_stack=decoder_stack,
        feed_forward_size=ff_size
    )(encoder_inputs, decoder_inputs)
    outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)
    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)


def train_model(vocab_size=512, max_len=128, batch_size=128, epochs=1):

    train, test = load_dataset(vocab_size, max_len)

    x_train, x_train_masks, y_train = train
    x_test, x_test_masks, y_test = test

    model = build_model(vocab_size, max_len)
    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit([x_train, x_train_masks], y_train,
              batch_size=batch_size, epochs=epochs, steps_per_epoch=np.math.ceil(len(x_train)/batch_size), callbacks=[es])

    # test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    # print("loss on Test: %.4f" % test_metrics[0])
    # print("accu on Test: %.4f" % test_metrics[1])


if __name__ == '__main__':
    train_model()