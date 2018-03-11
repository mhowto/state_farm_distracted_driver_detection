from starter import *
from vgg16 import Vgg16

import os, time, glob
from functools import reduce

import numpy as np
from tqdm import tqdm
import pickle
import tracemalloc
import gc
import sys

use_cache = 1 

train_x_pattern = 'train_r_{}_c_{}_t_{}_class_{}_x'
train_y_pattern = 'train_r_{}_c_{}_t_{}_class_{}_y'
train_id_pattern = 'train_r_{}_c_{}_t_{}_class_{}_id.dat'
unique_drivers_file = 'unique_dirvers.dat'

test_x_pattern = 'test_r_{}_c_{}_t_{}_x'
test_id_pattern = 'test_r_{}_c_{}_t_{}_id.dat'

cache_train_flag = 'cached_train.lock'
cache_test_flag = 'cached_test.lock'


def cache_train(raw_data, img_rows, img_cols, color_type, cl):
    X_train, y_train, X_train_id = raw_data
    np.save(os.path.join('cache', train_x_pattern.format(img_rows, img_cols, color_type, cl)), np.array(X_train))
    np.save(os.path.join('cache', train_y_pattern.format(img_rows, img_cols, color_type, cl)), np.array(y_train))
    path = os.path.join('cache', train_id_pattern.format(img_rows, img_cols, color_type, cl))
    with open(path, 'wb') as f:
        pickle.dump(X_train_id, f)

def restore_train(img_rows, img_cols, color_type, cl):
    X_train = np.load(os.path.join('cache', train_x_pattern.format(img_rows, img_cols, color_type, cl)+'.npy'))
    y_train = np.load(os.path.join('cache', train_y_pattern.format(img_rows, img_cols, color_type, cl)+'.npy'))
    with open(os.path.join('cache', train_id_pattern.format(img_rows, img_cols, color_type, cl)), 'rb') as f:
        X_train_id = pickle.load(f)
    return (X_train, y_train, X_train_id)

def cache_test(raw_data, img_rows, img_cols, color_type):
    X_test, test_id = raw_data
    np.save(os.path.join('cache', test_x_pattern.format(img_rows, img_cols, color_type)), np.array(X_test))
    with open(os.path.join('cache', test_id_pattern.format(img_rows, img_cols, color_type)), 'wb') as f:
        pickle.dump(test_id, f)

def restore_test(img_rows, img_cols, color_type):
    X_test = np.load(os.path.join('cache', test_x_pattern.format(img_rows, img_cols, color_type)+'.npy'))
    with open(os.path.join('cache', test_id_pattern.format(img_rows, img_cols, color_type)), 'r') as f:
        test_id = pickle.load(f)
    return X_test, test_id

def restore_all_train(img_rows, img_cols, color_type):
    x,y,train_id = [],[], []
    for cl in range(10):
        X_train, y_train, X_train_id = restore_train(img_rows, img_cols, color_type, cl)
        x.append(X_train)
        y.append(y_train)
        train_id.append(X_train_id)
    X_train = np.concatenate(x, axis=0)
    y_train = np.concatenate(y, axis=0)
    X_train_id = reduce(lambda head,tail: head+tail, train_id)
    with open(os.path.join('cache', unique_drivers_file), 'rb') as f:
        unique_drivers = pickle.load(f)
    return X_train, y_train, X_train_id, unique_drivers

def load_and_cache_train(img_rows, img_cols, color_type=1, path=DEFAULT_TRAIN_DATA_PATH):
    img2driver = get_driver_data()
    start_time = time.time()
    X_train_group = None
    y_train_group = None
    driver_id_group = []

    print('Read train images')
    for j in range(10):
        X_train, y_train = [], []
        X_train_id = []

        file_pattern = os.path.join(path, 'c'+str(j), '*.jpg')
        print('Load folder {}'.format(file_pattern))
        for fl in tqdm(glob.glob(file_pattern)):
            X_train_id.append(os.path.basename(fl))
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)

        if X_train_group is None:
            X_train_group = X_train
        else:
            X_train_group = np.concatenate((X_train_group, X_train), axis=0)

        if y_train_group is None:
            y_train_group = y_train
        else:
            y_train_group = np.concatenate((y_train_group, y_train), axis=0)
        
        driver_id = [img2driver[img_name] for img_name in X_train_id]
        driver_id_group.append(driver_id)

        cache_train([X_train, y_train, driver_id], img_rows, img_cols, color_type, j)

    driver_id = reduce(lambda x,y:x+y, driver_id_group)
    unique_drivers = sorted(list(set(driver_id)))

    path = os.path.join('cache', unique_drivers_file)
    with open(path, 'wb') as f:
        pickle.dump(unique_drivers, f)
    with open(os.path.join('cache', cache_train_flag), 'w') as f:
        f.write('1')
    
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train_group, y_train_group, driver_id, unique_drivers

def load_and_cache_test(img_rows, img_cols, color_type=1, path=DEFAULT_TEST_DATA_PATH):
    X_test, X_test_id = [], []
    start_time = time.time()
    print('Read test images')
    file_pattern = os.path.join(path, '*.jpg')
    for fl in tqdm(glob.glob(file_pattern)):
        X_test_id.append(os.path.basename(fl))
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img)

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    cache_test((X_test, X_test_id), img_rows, img_cols, color_type)
    with open(os.path.join('cache', cache_test_flag), 'w') as f:
        f.write('1')
    return X_test, X_test_id

def read_and_normalize_train_data2(img_rows, img_cols, color_type=1):
    cached_flag = None
    cache_file_path = os.path.join('cache', cache_train_flag)
    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'r') as f:
            cached_flag = f.read()
    if cached_flag is None or int(cached_flag) != 1:
        train_data, train_target, driver_id, unique_drivers = load_and_cache_train(img_rows, img_cols, color_type)
    else:
        print('Restore train from cache!')
        train_data, train_target, driver_id, unique_drivers = restore_all_train(img_rows, img_cols, color_type)
    
    """
    train_data = np.array(train_data, dtype='f')
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = np.reshape(train_data, (train_data.shape[0], color_type, img_rows, img_cols))
    train_target = to_categorical(train_target)
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train_samples')
    """
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data2(img_rows, img_cols, color_type=1):
    cached_flag = None
    cache_file_path = os.path.join('cache', cache_test_flag)
    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'r') as f:
            cached_flag = f.read()
    if cached_flag is None or int(cached_flag) != 1:
        test_data, test_id = load_and_cache_test(img_rows, img_cols, color_type)
    else:
        print('Restore train from cache!')
        test_data, test_id = restore_test(img_rows, img_cols, color_type)
    
    return test_data, test_id

class Vgg16V1(object):
    def __init__(self, img_rows, img_cols, color_type, learning_rate=0.001, data_format='channels_last'):
        self.data_format = data_format

        self.images = tf.placeholder(tf.float32, [None, img_rows, img_cols, 3], name='images')
        self.labels =  tf.placeholder(tf.uint8, [None, 10], name='labels')

        self.vgg16 = Vgg16('data/vgg16.npy')
        self.vgg16.build(self.images)

        self.outputs = tf.layers.dense(self.vgg16.relu7, 10, activation=tf.nn.softmax)
        self.loss = tf.losses.softmax_cross_entropy(self.labels, logits=self.outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def fit(self, sess, x, y, batch_size=1024, nb_epoch=20):
        it = 0
        for _x, _y in BatchIterator((x, y), batch_size=batch_size, epoch=nb_epoch):
            # if self.data_format == 'channels_first':
                # _x = np.transpose(_x, [0,1,2,3])
            it += 1
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.images: _x, self.labels: _y})
            if it % 100 == 0:
                print('Iter {} loss: {}'.format(it, loss))
    
    def predict(self, sess, x, batch_size=1024):
        predicts = []
        for _x in BatchIterator(x, batch_size=batch_size, mode='test'):
            # if self.data_format == 'channels_first':
                # _x = np.transpose(_x, [0,2,3,1])
            logits = sess.run(self.outputs, feed_dict={self.images: _x})
            predicts.append(logits)
        return np.concatenate(predicts, axis=0)
     

def run_transfer_single():
    #input image dimensions
    img_rows, img_cols = 224, 224
    batch_size = 32
    nb_epoch = 20
    random_state = 51
    color_type = 3

    tracemalloc.start()

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data2(img_rows, img_cols, color_type=color_type)
    train_target = to_categorical(train_target)
    print('train_data shape:', train_data.shape)
    print('train_target shape:', train_target.shape)
    print('dirver_id shape:', len(driver_id))
    print('unique_drivers shape:', len(unique_drivers))

    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
    # unique_list_train = ['p002', 'p012', 'p014']
    X_train, Y_train = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    unique_list_valid = ['p081']
    X_valid, Y_valid = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)
    print('X_valid shape:', X_valid.shape)
    print('Y_valid shape:', Y_valid.shape)

    del train_data
    del train_target
    # gc.collect()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

    with tf.Graph().as_default():
        vgg16v1 = Vgg16V1(img_rows, img_cols, color_type, data_format='channels_first')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            vgg16v1.fit(sess, X_train, Y_train, batch_size=batch_size)
            predictions_valid = vgg16v1.predict(sess, X_valid)
            loss = log_loss(Y_valid, predictions_valid)
            print('validate loss: {}'.format(loss))
            save_path = saver.save(sess, 'model/vgg16v1_model_{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

    '''
    test_data, test_id = read_and_normalize_test_data2(img_rows, img_cols, color_type=color_type)
    print('test_data shape:', test_data.shape)
    print('test_id shape:', test_id.shape)
    '''


if __name__ == '__main__':
    run_transfer_single()