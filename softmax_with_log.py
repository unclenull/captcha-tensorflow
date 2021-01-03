# -*- coding:utf-8 -*-
import argparse
import datetime
import sys
import tensorflow as tf

import datasets.base as input_data


MAX_STEPS = 10000
BATCH_SIZE = 100

LOG_DIR = 'log/regression-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)


def main(_):
    # load data
    meta, train_data, test_data = input_data.load_data(FLAGS.data_dir, flatten=True)
    print('data loaded. train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))

    LABEL_SIZE = meta['label_size']
    IMAGE_WIDTH = meta['width']
    IMAGE_HEIGHT = meta['height']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    # variable in the graph for input data
    with tf.compat.v1.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, LABEL_SIZE])
        variable_summaries(x)
        variable_summaries(y_)

        # must be 4-D with shape `[batch_size, height, width, channels]`
        images_shaped_input = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.compat.v1.summary.image('input', images_shaped_input, max_outputs=LABEL_SIZE*2)

    # define the model
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.compat.v1.name_scope('linear_model'):
        with tf.compat.v1.name_scope('W'):
            W = tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]))
            variable_summaries(W)
        with tf.compat.v1.name_scope('b'):
            b = tf.Variable(tf.zeros([LABEL_SIZE]))
            variable_summaries(b)
        with tf.compat.v1.name_scope('y'):
            y = tf.matmul(x, W) + b
            tf.compat.v1.summary.histogram('y', y)

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.
    with tf.compat.v1.name_scope('loss'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_), logits=y)
        cross_entropy = tf.reduce_mean(input_tensor=diff)
        train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        variable_summaries(diff)

    # forword prop
    predict = tf.argmax(input=y, axis=1)
    expect = tf.argmax(input=y_, axis=1)

    # evaluate accuracy
    with tf.compat.v1.name_scope('evaluate_accuracy'):
        correct_prediction = tf.equal(predict, expect)
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
        variable_summaries(accuracy)

    with tf.compat.v1.Session() as sess:

        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + '/train', sess.graph)

        tf.compat.v1.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            train_summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            train_writer.add_summary(train_summary, i)

            if i % 100 == 0:
                # Test trained model
                test_summary, r = sess.run([merged, accuracy], feed_dict={x: test_data.images, y_: test_data.labels})
                train_writer.add_summary(test_summary, i)
                print('step = %s, accuracy = %.2f%%' % (i, r * 100))

        train_writer.close()

        # final check after looping
        test_summary, r_test = sess.run([merged, accuracy], feed_dict={x: test_data.images, y_: test_data.labels})
        train_writer.add_summary(test_summary, i)
        print('testing accuracy = %.2f%%' % (r_test * 100, ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/char-1-epoch-2000/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
