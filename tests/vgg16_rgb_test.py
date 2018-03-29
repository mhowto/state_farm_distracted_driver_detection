import unittest

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgg16_rgb import Vgg16

import numpy as np
import tensorflow as tf


io_dir = 'vgg16_io_data'

def generate_data_by_keras():
    from keras.applications import VGG16
    v16 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))


    if not os.path.exists(io_dir):
        os.mkdir(io_dir)

    input = np.random.uniform(0.0, 255.0, [10, 224, 224, 3])
    output = v16.predict(input)

    with open(os.path.join(io_dir, 'input'), 'wb') as f:
        np.save(f, input)
    with open(os.path.join(io_dir, 'output'), 'wb') as f:
        np.save(f, output)

generate_data_by_keras()


class TestVgg16(unittest.TestCase):
    def test_io(self):
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg16 = Vgg16('../data/vgg16.npy')
        vgg16.build(images)

        inputs = np.load(os.path.join(io_dir, 'input'))
        outputs = np.load(os.path.join(io_dir, 'output'))
        with tf.Session() as sess:
            prob = sess.run(vgg16.prob, feed_dict={images: inputs})
        self.assertEqual(outputs.shape, prob.shape)
        error = np.sum(np.square(prob - outputs))
        print(error)
        print(prob-outputs)
