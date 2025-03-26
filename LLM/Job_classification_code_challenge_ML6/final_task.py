#!/usr/bin/env python3

"""Train and export the model

This file trains the model upon all data with the arguments it got via
the gcloud command.
"""

import os
import argparse
import logging

from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf

import trainer.data as data
import trainer.model as model


def export_model(ml_model, export_dir, model_dir='exported_model'):
    """Prepare model for strings representing text data and export it.

    Before the model is exported the initial layers of the model need to be
    adapted to handle prediction of text contained in JSON files.

    Parameters:
        ml_model: A compiled model
        export_dir: A string specifying the
        model_dir: A string specifying the name of the directory to
            which the model is written.
    """
    prediction_input = tf.keras.Input(
        dtype=tf.string, name='bytes', shape=())

    prediction_output = ml_model(prediction_input)
    prediction_output = tf.keras.layers.Lambda(
        lambda x: x, name='PROBABILITIES')(prediction_output)
    prediction_class = tf.keras.layers.Lambda(
        lambda x: tf.argmax(x, 1), name='CLASSES')(prediction_output)
    
    ml_model = tf.keras.models.Model(prediction_input, outputs=[prediction_class, prediction_output])

    model_path = Path(export_dir) / model_dir
    if model_path.exists():
        timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
        model_path = Path(str(model_path) + timestamp)

    tf.saved_model.save(ml_model, str(model_path))


def train_and_export_model(params):
    """The function gets the training data from the training folder and
    the eval folder.
    Your solution in the model.py file is trained with this training data.
    The evaluation in this method is not important since all data was already
    used to train.

    Parameters:
        params: Parameters for training and exporting the model
    """
    (train_data, train_labels) = data.load_dataset("data/train.csv")
    (eval_data, eval_labels) = data.load_dataset("data/eval.csv")

    train_data = np.append(train_data, eval_data, axis=0)
    train_labels = np.append(train_labels, eval_labels, axis=0)

    input_layer = tf.keras.Input(shape=(), name='input_text', dtype=tf.string)

    ml_model = model.solution(input_layer)

    ml_model.fit(train_data, train_labels,
                 batch_size=model.get_batch_size(),
                 epochs=model.get_epochs())
    export_model(ml_model, export_dir=params.job_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default='output',
        help='directory to store checkpoints'
    )

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train_and_export_model(args)
