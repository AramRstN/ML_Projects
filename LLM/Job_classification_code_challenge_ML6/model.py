#!/usr/bin/env python3

"""Model to classify job descriptions

This file contains all the model information: the training steps, the batch
size and the model itself.
"""


import csv
import re
import string

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses


SEQUENCE_LENGTH = 1000
MAX_FEATURES = 10000
EMBEDDING_DIM = 64


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 128


def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 100


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


class TxtClassifier(keras.Model):
    def __init__(self):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Embedding(MAX_FEATURES, EMBEDDING_DIM),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(5, activation='sigmoid')
        ])

        self.vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=MAX_FEATURES,
            output_mode='int',
            output_sequence_length=SEQUENCE_LENGTH
            )

        with open('data/train.csv') as train_data:
            data_reader = csv.reader(train_data, delimiter=",", quotechar='"')
            next(data_reader)
            data = []
            for _, desc in data_reader:
                data.append(desc)
        self.vectorize_layer.adapt(np.array(data))

    def vectorize_text(self, text):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text)

    def call(self, inputs):
        return self.seq(self.vectorize_text(inputs))

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identify the different job 
    description classes. The model's outputs are expected to be probabilities 
    for the classes and and it should be ready for training.
    The input layer specifies the shape of the text. 

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            tf.string, shape: ()
    Returns:
        model: A compiled model
    """

    # TODO: Code of your solution
    model = TxtClassifier()
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer='adam',
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
        )
    # TODO: Return the compiled model
    return model
