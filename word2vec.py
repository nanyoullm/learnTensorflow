# -*- coding: UTF-8 -*-

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data('/mypar/Data/text8.zip')

vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    '''
    data: 每个单词的编号
    count: 单词和对于的编号列表
    dictionary: 单词和对应编号的字典
    reverse_dictionary: 字典翻转

    编号越小，出现频次越高
    '''
    return data, count, dictionary, reverse_dictionary


data_index = 0
data, count, dictionary, reverse_dictionary = build_dataset(words)


def generate_batch(batch_size, num_skips, skip_window):
    '''
    batch_size: batch的大小
    num_skips: 对每个单词生成的样本个数
    skip_window: 单词最远关联到的单词
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    # 先填充span个单词的buffer
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 在这里,buffer里面存着3个单词,[a,b,c],其中b为源单词,a和c为目标单词,生成的数据为[b,a],[b,c]
    for i in range(batch_size // num_skips):
        # 目标单词
        target = skip_window
        # 避免目标单词，用于生成样本
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print (batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
           reverse_dictionary[labels[i, 0]])

batch_size = 128
# 词向量的维度
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        # embedings为50000个样本的128维向量表示, 50000*128
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )

        # embed为本次训练数据的表示,初始值为随机
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 训练参数
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    print vocabulary_size, type(vocabulary_size)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalizer_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalizer_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalizer_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

    num_steps = 100001

with tf.Session(graph=graph) as sess:
    init.run()
    print 'initialized'

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 200
            print 'average loss at step {}: {}'.format(step, average_loss)
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k+1]
                log_str = 'Nearest to {}:'.format(valid_word)
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '{} {}'.format(log_str, close_word)
                print log_str
        final_embedding = normalizer_embeddings.eval()