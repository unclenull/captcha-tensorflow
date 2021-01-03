# -*- coding:utf-8 -*-
import argparse
import sys
import tensorflow as tf

import datasets.base as input_data


MAX_STEPS = 10000
BATCH_SIZE = 1000

FLAGS = None


def main(_):
    # load data
    meta, train_data, test_data = input_data.load_data(FLAGS.data_dir, flatten=True)
    print('data loaded')
    print('train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))

    LABEL_SIZE = meta['label_size']
    IMAGE_SIZE = meta['width'] * meta['height']
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    # variable in the graph for input data
    x = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, LABEL_SIZE])

    # define the model
    W = tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]))
    b = tf.Variable(tf.zeros([LABEL_SIZE]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_), logits=y)
    cross_entropy = tf.reduce_mean(input_tensor=diff)
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # forword prop
    predict = tf.argmax(input=y, axis=1)
    expect = tf.argmax(input=y_, axis=1)

    # evaluate accuracy
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 100 == 0:
                # Test trained model
                r = sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels})
                print('step = %s, accuracy = %.2f%%' % (i, r * 100))

        # final check after looping
        r_test = sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels})
        print('testing accuracy = %.2f%%' % (r_test * 100, ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/char-1-epoch-2000/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
