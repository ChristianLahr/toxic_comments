import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from keras.layers import GRU, CuDNNGRU, Dropout, Bidirectional, BatchNormalization, SpatialDropout1D
from toxic_comments.used_cleaned.global_variables import GPU_MODEL, NUMBER_OF_CLASSES

class CNN:

    pass


class CCNN:

    pass


class CRNN:

    pass

class BIRNN:

    @staticmethod
    def gru64_3(embedding_matrix, x, keep_prob):
        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        if GPU_MODEL:
            x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
            x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        else:
            x2 = Bidirectional(GRU(64, return_sequences=True))(embedded_input)
            x2 = Bidirectional(GRU(64, return_sequences=True))(x2)
        outputs = tf.transpose(x2, [0, 2, 1])

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)
        x3 = tf.nn.dropout(x3, keep_prob=keep_prob)
        # outputs = BatchNormalization()(x2)
        # x3 = layers.fully_connected(outputs, NUMBER_OF_CLASSES*2, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, NUMBER_OF_CLASSES, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def gru64_4(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        if GPU_MODEL:
            x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
        else:
            x2 = Bidirectional(GRU(64, return_sequences=True))(embedded_input)
        outputs = tf.transpose(x2, [0, 2, 1])

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)
        x3 = tf.nn.dropout(x3, keep_prob=keep_prob)
        # outputs = BatchNormalization()(x2)
        # x3 = layers.fully_connected(outputs, NUMBER_OF_CLASSES*2, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, NUMBER_OF_CLASSES, activation_fn=tf.nn.sigmoid)
        return logits

class CCAPS:

    pass

class CAPS:

    pass

class DENSE:

    pass

class CNNRNN:

    pass

class HYBRID:

    pass