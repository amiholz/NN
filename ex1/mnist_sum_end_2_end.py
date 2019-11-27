# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def define_network(x, keep_prob):
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28 * 2, 28, 1])

    with tf.name_scope("conv_1"):
        W_conv_1 = weight_variable([5, 5, 1, 32])
        b_conv_1 = bias_variable([32])
        h_conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)

    with tf.name_scope("pool_1"):
        h_pool_1 = max_pool_2x2(h_conv_1)

    with tf.name_scope("conv_2"):
        W_conv_2 = weight_variable([5, 5, 32, 64])
        b_conv_2 = bias_variable([64])
        h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)

    with tf.name_scope("pool_2"):
        h_pool_2 = max_pool_2x2(h_conv_2)

    with tf.name_scope("fc_1"):
        # W_fc_1 = weight_variable([28 * 28, 1024])
        # b_fc_1 = bias_variable([1024])
        # h_pool2_flat = tf.reshape(x_image, [-1, 28 * 28])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc_1) + b_fc_1)
        W_fc_1 = weight_variable([7 * 14 * 64, 1024])
        b_fc_1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool_2, [-1, 7 * 14 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc_1) + b_fc_1)

    with tf.name_scope('dropout'):
        h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc_2"):
        W_fc_2 = weight_variable([1024, 1])
        b_fc_2 = bias_variable([1])
        y_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2

    return y_conv


def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()
    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784 * 2], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, 2], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    y = define_network(x, keep_prob)

    with tf.name_scope('loss'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.losses.sparse_softmax_cross_entropy on the
        # raw logit outputs of the nn_layer above, and then average across
        # the batch.
        with tf.name_scope('total'):
            loss = tf.reduce_mean(tf.abs(tf.cast((y_[:, 0] + y_[:, 1]), tf.float32) - y))

    tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.abs(y - tf.cast(y_[:, 0] + y_[:, 1], tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        assert xs.shape[0] % 2 == 0
        xs = np.c_[xs[:xs.shape[0] // 2, :], xs[xs.shape[0] // 2:, :]]
        ys = np.c_[ys[:ys.shape[0] // 2], ys[ys.shape[0] // 2:]]
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 500 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=5001,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/mnist_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
