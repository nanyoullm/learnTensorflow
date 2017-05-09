# -*- coding: UTF-8 -*-
# 使用rnn预测mnist

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
mnist = input_data.read_data_sets('/mypar/Data/mnist/', one_hot=True)

# hyperparamters
lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

weigths = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# 定义rnn结构
def RNN(X, weights, biases):
    # 原始X的维度为[batch_size: 128, n_steps: 28, n_inputs: 28]
    # 进入rnn前有一层dense layer, 故先将X转为2维 --> [128*28, 28]
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 再将X_in转回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 定义rnn的cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                             forget_bias=1.0,
                                             state_is_tuple=True)
    # cell里的状态初始化
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # cell中的计算
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state)

    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results

pred = RNN(x, weights=weigths, biases=biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 训练rnn
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('/tmp/lstm_mnist', sess.graph)
    step = 0

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op],
                 feed_dict={
                     x: batch_xs,
                     y: batch_ys
                 })
        if step % 20 == 0:
            print sess.run([accuracy],
                           feed_dict={
                               x: batch_xs,
                               y: batch_ys
                           })
        step += 1