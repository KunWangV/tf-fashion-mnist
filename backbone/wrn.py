# coding: utf-8

import tensorflow as tf


def se_layer(inputs, r=8):
    c = inputs.get_shape().as_list()[-1]
    k = inputs.get_shape().as_list()[-2]
    avg = tf.layers.average_pooling2d(inputs, k, strides=1)
    down = tf.layers.conv2d(avg, c // r, kernel_size=1, strides=1)
    relu1 = tf.nn.relu(down)
    up = tf.layers.conv2d(relu1, c, kernel_size=1, strides=1)
    out = tf.sigmoid(up)
    return out


def basic_block(inputs, in_planes, out_planes, stride, drop_rate=0.5, training=False, name=""):
    # with tf.name_scope(name) as scope:
    skip_conv = in_planes != out_planes

    bn0 = tf.layers.batch_normalization(inputs, training=training)
    relu0 = tf.nn.relu(bn0)

    conv1 = tf.layers.conv2d(relu0, out_planes, strides=stride, kernel_size=3, padding='same', )
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    if training:
        relu1 = tf.layers.dropout(relu1, drop_rate)

    res = tf.layers.conv2d(relu1, out_planes, strides=1, kernel_size=3, padding='same', )

    # se block
    se_out = se_layer(res)
    res = se_out * res

    match_conv = None
    if skip_conv:
        match_conv = tf.layers.conv2d(relu0, out_planes, strides=stride, kernel_size=3, padding='same', )

    return tf.add(res, match_conv if skip_conv else inputs)


def network_block(inputs, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, name='', training=False):
    for i in range(nb_layers):
        outputs = block(inputs, in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate,
                        name='{}/{}'.format(name, i), training=training)
        inputs = outputs

    return inputs


def wide_resnet(inputs, training, depth, num_classes, wide_factor=1, drop_rate=0.5):
    """
    wide resent model
    :param inputs:
    :param training: bool
    :param depth:
    :param num_classes:
    :param wide_factor: expand factor
    :param drop_rate: drop rate for dropout layer
    :return:
    """
    n_channels = [16, 16 * wide_factor, 32 * wide_factor, 64 * wide_factor]
    n = (depth - 4) // 6

    with tf.name_scope("head"):
        conv1 = tf.layers.conv2d(inputs, n_channels[0], (3, 3), strides=(1, 1), padding='same',
                                 )

    with tf.name_scope('b1'):
        block1 = network_block(conv1, n, n_channels[0], n_channels[1], basic_block, 1, drop_rate, 'b1')

    with tf.name_scope('b2'):
        block2 = network_block(block1, n, n_channels[1], n_channels[2], basic_block, 2, drop_rate, 'b2')

    with tf.name_scope('b3'):
        block3 = network_block(block2, n, n_channels[2], n_channels[3], basic_block, 2, drop_rate, 'b3')

    with tf.name_scope('fc'):
        fc_bn = tf.layers.batch_normalization(block3, training=training)
        fc_ac = tf.nn.relu(fc_bn)
        fc_pool = tf.layers.average_pooling2d(fc_ac, 7, 1)
        fc_out = tf.layers.dense(tf.layers.flatten(fc_pool), num_classes,
                                 )

    return fc_out


def wrn40_4(inputs, mode=False):
    if type(mode) == str:
        mode = mode == tf.estimator.ModeKeys.TRAIN

    return wide_resnet(inputs, mode, 40, 10, 4)


def wrn28_4(inputs, mode=False):
    if type(mode) == str:
        mode = mode == tf.estimator.ModeKeys.TRAIN

    return wide_resnet(inputs, mode, 28, 10, 4)


def wrn28_10(inputs, mode=False):
    if type(mode) == str:
        mode = mode == tf.estimator.ModeKeys.TRAIN

    return wide_resnet(inputs, mode, 28, 10, 10)


if __name__ == '__main__':
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    wide_resnet(inputs, True, 40, 10, 4)
