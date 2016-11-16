from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import numpy as np
import scipy.misc


def decode_one_hot(vector):
    return [i for i, x in enumerate(vector) if x == 1]


def print_image(n, mnist):
    out = 1 - mnist.train.images[n]
    out = np.reshape(out, (28, 28))
    scipy.misc.imsave('outfile.jpg', out)
    print(decode_one_hot(mnist.train.labels[n]))


def create_model(mnist):
    # we can input here any MNIST pic, None means a dimension can be of any length
    x = tf.placeholder(tf.float32, [None, 784])
    # Variable - modifable tensor
    # Weights full of zeros pixels x classes
    W = tf.Variable(tf.zeros([784, 10]))
    # bias full of zeros
    b = tf.Variable(tf.zeros([10]))
    # our evidence table WITHOUT SOFTMAX!
    y = tf.matmul(x, W) + b
    # placeholder for crossentropy
    y_ = tf.placeholder(tf.float32, [None, 10])
    # softmax cross entropy between logits and labels (y and y_)
    # logits: Unscaled log probabilities.
    # we get one value. entropy == 0 means there is no doubt
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # minimalizing cross entropy
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # create the operation to initialize
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_xs})
    # tf argmax gives the index of highest enty
    # here argmax returns the label that our model thinks is most probable
    # we get a list of booleans - which images were correctly
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # print_image(1, mnist)
    create_model(mnist)
