"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  # NOTE: should add $GRAPE_DATASET_DIR before your original data path
  data_env = os.environ.get('GRAPE_DATASET_DIR')
  data_path = os.path.join(data_env, FLAGS.data_dir)
  mnist = input_data.read_data_sets(data_path, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train

  # NOTE: mnist training images number is 60000
  num_train_img = 1000
  total_img = num_train_img * FLAGS.num_epochs
  epoch_count = 0
  for i in range(total_img):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % num_train_img == 0:
        epoch_count += 1
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        valid_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        # NOTE: Output format of learning curve
        print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
                (epoch_count, train_accuracy, epoch_count, valid_accuracy))
        # NOTE: Save models to outdir
        if not os.path.isdir(FLAGS.outdir):
            os.mkdir(FLAGS.outdir)
        save_path = os.path.join(FLAGS.outdir, "epoch-"+str(epoch_count)+".ckpt")
        saver.save(sess, save_path)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory for storing input data')
  parser.add_argument('--num-epochs', type=int, default=1,
                      help='Number of epochs')
  parser.add_argument('--outdir', type=str, default='./model/',
                      help='model destination')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

