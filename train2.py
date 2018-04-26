# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

DATA_DIR = './fashion-mnist/data/fashion'

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 300
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
TEST_FREQUENCY = 2000

import numpy as np

tf.app.flags.DEFINE_string('model', 'base_model', "model name")
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('optimizer', 'sgd', "sgd, adam")
tf.app.flags.DEFINE_integer('batch_size', 400, 'batch size')
tf.app.flags.DEFINE_string('model_dir', "/tmp/mnist_convnet_model", 'model to save')

tf.app.flags.DEFINE_boolean('use_fp16', False, 'fp 16')
tf.app.flags.DEFINE_boolean('self_test', False, 'self testF')

FLAGS = tf.app.flags.FLAGS

import models


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


model = getattr(models, FLAGS.model)


def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = numpy.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
    for image in range(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) /
                    predictions.shape[0])


def main(_):
    # Extract it into numpy arrays.
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    train_data = train_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

    test_data = mnist.test.images  # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    test_data = test_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    # test_data, test_labels = shuffle(test_data, test_labels)

    # train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    # test_data = (test_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        data_type(),
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        data_type(),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))

    # L2 regularization for the fully connected parameters.
    # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # # Add the regularization term to the loss.
    # loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        FLAGS.lr,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # Create a local session to run the training.
    start_time = time.time()
    best = 1.0
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print('Initialized!')
        # Loop through training steps.
        for step in range(int(num_epochs * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                val_error = error_rate(eval_in_batches(validation_data, sess), validation_labels)

                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY), 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr),
                      'Validation error: %.1f%%' % val_error)
                sys.stdout.flush()

            if step % TEST_FREQUENCY == 0:
                # Finally print the result!
                test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
                if test_error < best:
                    best = test_error
                print('Test error: %.1f%%' % test_error)
                if FLAGS.self_test:
                    print('test_error', test_error)
                    assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                        test_error,)

        print('best: ', best)


if __name__ == '__main__':
    tf.app.run(main=main)
