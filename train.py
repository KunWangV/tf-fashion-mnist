from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from skimage.transform import rescale, resize
from transform import seq_transform, batch_random_erase

import models

tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './fashion-mnist/data/fashion'

tf.app.flags.DEFINE_string('model', 'base_model', "model name")
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('optimizer', 'sgd', "sgd, adam")
tf.app.flags.DEFINE_integer('batch_size', 400, 'batch size')
tf.app.flags.DEFINE_string('model_dir', "/tmp/mnist_convnet_model", 'model to save')

FLAGS = tf.app.flags.FLAGS


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # input_layer = features['x']
    logits = getattr(models, FLAGS.model)(input_layer, mode)  # get model

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def data_generator(x, y, batch_size):
    x = np.reshape(x, (-1, 28, 28, 1))
    # x = seq_transform.augment_images(x)
    # x = batch_random_erase(x)
    print('transform done!')
    dataset = tf.data.Dataset.from_tensor_slices(({"x": x}, y))
    dataset = dataset.shuffle(20000).repeat(10).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def main(unused_argv):
    # Load training and eval data
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    print(train_data.shape, eval_data.shape)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for j in range(300):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
