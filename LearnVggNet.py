# -*- coding: UTF-8 -*-
# 学习VGGNet网络,并测试其训练速度

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


def conv_op(input_op, name, kernel_h, kernel_w, n_out, stride_h, stride_w, paramters):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name=scope+'_weights',
            shape=[kernel_h, kernel_w, n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )

        conv = tf.nn.conv2d(input_op, kernel, strides=[1, stride_h, stride_w, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(name=scope+'_biases', initializer=bias_init_val, trainable=True)
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)

        paramters += [kernel, biases]

        return activation


def fc_op(input_op, name, n_out, paramters):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+'_weights',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name=scope+'_biases')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)

        paramters += [kernel, biases]

        return activation


def max_pool_op(input_op, name, kernel_h, kernel_w, stride_h, stride_w):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kernel_h, kernel_w, 1,],
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name=name)


# 构建网络结构graph
def inference_op(input_op, keep_prob):
    paramters = []

    conv1_1 = conv_op(input_op, 'conv1_2', 3, 3, 64, 1, 1, paramters)
    conv1_2 = conv_op(conv1_1, 'conv1_2', 3, 3, 64, 1, 1, paramters)
    pool1 = max_pool_op(conv1_2, 'pool1', 2, 2, 2, 2)

    conv2_1 = conv_op(pool1, 'conv2_1', 3, 3, 128, 1, 1, paramters)
    conv2_2 = conv_op(conv2_1, 'conv2_2', 3, 3, 128, 1, 1, paramters)
    pool2 = max_pool_op(conv2_2, 'pool2', 2, 2, 2, 2)

    conv3_1 = conv_op(pool2, 'conv3_1', 3, 3, 256, 1, 1, paramters)
    conv3_2 = conv_op(conv3_1, 'conv3_2', 3, 3, 256, 1, 1, paramters)
    conv3_3 = conv_op(conv3_2, 'conv3_3', 3, 3, 256, 1, 1, paramters)
    pool3 = max_pool_op(conv3_3, 'pool3', 2, 2, 2, 2)

    conv4_1 = conv_op(pool3, 'conv4_1', 3, 3, 512, 1, 1, paramters)
    conv4_2 = conv_op(conv4_1, 'conv4_2', 3, 3, 512, 1, 1, paramters)
    conv4_3 = conv_op(conv4_2, 'conv4_3', 3, 3, 512, 1, 1, paramters)
    pool4 = max_pool_op(conv4_3, 'pool4', 2, 2, 2, 2)

    conv5_1 = conv_op(pool4, 'conv4_1', 3, 3, 512, 1, 1, paramters)
    conv5_2 = conv_op(conv5_1, 'conv4_2', 3, 3, 512, 1, 1, paramters)
    conv5_3 = conv_op(conv5_2, 'conv4_3', 3, 3, 512, 1, 1, paramters)
    pool5 = max_pool_op(conv5_3, 'pool5', 2, 2, 2, 2)

    # tensor向量化
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    fc6 = fc_op(resh1, 'fc6', 4096, paramters)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, paramters=paramters)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, paramters=paramters)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fc8, paramters


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_steps_burn_in + num_batches):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time

        if i >= num_steps_burn_in:
            if not i % 10:
                print '{}: step {}, duration = {}'.format(datetime.now(), i, duration)
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)

    print '{}: {} across {} steps, {} +/- {} sec per batch'.format(datetime.now(), info_string, num_batches, mn, sd)


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        image = tf.Variable(tf.random_normal([batch_size,
                                              image_size,
                                              image_size,
                                              3],
                                             dtype=tf.float32,
                                             stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(image, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter('/tmp/vggnet', sess.graph)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, 'Forward')

        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, 'Forward-Backward')


run_benchmark()