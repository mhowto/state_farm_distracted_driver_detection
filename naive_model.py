import tensorflow as tf
from batch_iterator import BatchIterator
import numpy as np

class NaiveModel(object):
    def __init__(self, img_rows, img_cols, num_classes, color_type=1, learning_rate=0.001, name='naive_model'):
        self.images = tf.placeholder(tf.float32, [None, color_type, img_rows, img_cols], name='images')
        self.labels =  tf.placeholder(tf.float32, [None, num_classes], name='labels')
        self.prob = tf.placeholder_with_default(0.5, shape=())

        with tf.variable_scope(name):
            self.conv1 = tf.layers.conv2d(self.images, 32, kernel_size=[3, 3], activation=tf.nn.relu, data_format='channels_first')
            self.max_pooling1 = tf.layers.max_pooling2d(self.conv1, pool_size=[2,2], strides=2, data_format='channels_first')
            self.dropout1 = tf.layers.dropout(self.max_pooling1, self.prob)

            self.conv2 = tf.layers.conv2d(self.dropout1, 64, [3,3], activation=tf.nn.relu, data_format='channels_first')
            self.max_pooling2 = tf.layers.max_pooling2d(self.conv1, pool_size=[2,2], strides=2, data_format='channels_first')
            self.dropout2 = tf.layers.dropout(self.max_pooling2, self.prob)

            self.conv3 = tf.layers.conv2d(self.dropout2, 128, [3,3], activation=tf.nn.relu, data_format='channels_first')
            self.max_pooling3 = tf.layers.max_pooling2d(self.conv3, [8,8], 8, data_format='channels_first')
            self.dropout3 = tf.layers.dropout(self.max_pooling3, self.prob)

            self.flatten = tf.layers.flatten(self.dropout3)
            self.outputs = tf.layers.dense(self.flatten, num_classes, activation=tf.nn.softmax)

            self.loss = tf.losses.softmax_cross_entropy(self.labels, logits=self.outputs)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
    
    def fit(self, sess, x, y, batch_size=1024, nb_epoch=20):
        it = 0
        for _x, _y in BatchIterator((x, y), batch_size=batch_size, epoch=nb_epoch):
            it += 1
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.images: _x, self.labels: _y, self.prob: 0.5})
            if it % 100 == 0:
                print('Iter {} loss: {}'.format(it, loss))
    
    def predict(self, sess, x, batch_size=1024):
        predicts = []
        for _x in BatchIterator(x, batch_size=batch_size, mode='test'):
            logits = sess.run(self.outputs, feed_dict={self.images: _x, self.prob: 1.0})
            predicts.append(logits)
        return np.concatenate(predicts, axis=0)
