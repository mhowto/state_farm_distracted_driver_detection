from starter import *
from vgg16 import Vgg16

import os, time, glob
from functools import reduce

import numpy as np
from tqdm import tqdm
from sklearn.cross_validation import train_test_split

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

def get_minibatches(x, y, minibatch_size, shuffle=True):
    '''
    Genreate the mini batches
    '''
    data_size = x.shape[0]
    indicies = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indicies)
    for start_idx in range(0, data_size, minibatch_size):
        idx = indicies[start_idx : start_idx + minibatch_size]
        if y is not None:
            yield x[idx], y[idx]
        else:
            yield x[idx]

class Vgg16V1(object):
    def __init__(self, img_rows, img_cols, color_type, learning_rate=0.001, data_format='channels_last'):
        self.data_format = data_format

        self.images = tf.placeholder(tf.float32, [None, img_rows, img_cols, 3], name='images')
        self.labels = tf.placeholder(tf.uint8, [None, 10], name='labels')

        self.vgg16 = Vgg16('data/vgg16.npy')
        self.vgg16.build(self.images)

        #self.outputs = tf.layers.dense(self.vgg16.relu7, 10, activation=tf.nn.softmax, kernel_initializer=tf.truncated_normal_initializer)
        self.outputs = tf.layers.dense(self.vgg16.relu7, 10, activation=tf.nn.softmax)
        self.loss = tf.losses.softmax_cross_entropy(self.labels, logits=self.outputs)
        self.correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(self.loss)

    def train_on_batch(self, sess, inputs_batch, labels_batch, is_training=True, dropout=1):
        # feed = self.create_feed_dict(inputs_batch, labels_batch, is_training)
        variables = [self.loss, self.correct_prediction, self.optimizer]
        if not is_training:
            variables[-1] = self.accuracy
        loss, corr, _ = sess.run([self.loss, self.correct_prediction, self.optimizer],
            feed_dict={self.images: inputs_batch, self.labels: labels_batch})
        return loss, corr

    def run_epoch(self, sess, batch_size, training_set, validation_set, dropout):
        X_tr, Y_tr = training_set
        X_val, Y_val = validation_set

        with tqdm(get_minibatches(X_tr, Y_tr, batch_size, False)) as pbar:
            for train_x, train_y in pbar:
                loss, corr = self.train_on_batch(sess, train_x, train_y, True, dropout)
                pbar.set_description('train_loss: {0:.3g}, train_acc: {1:.3g}'.format(loss, np.sum(corr) / train_x.shape[0]))

        val_loss, val_corr = 0, 0
        with tqdm(get_minibatches(X_val, Y_val, batch_size, False)) as pbar:
            for val_x, val_y in pbar:
                loss, corr = self.train_on_batch(sess, val_x, val_y, False)
                val_loss += loss
                val_corr += np.sum(corr)
                pbar.set_description('val_loss: {0:.3g}, val_acc: {1:.3g}'.format(loss, np.sum(corr) / val_x.shape[0]))
            print("Validation loss = {0:.3g} and accuracy = {1:.3g}".format(val_loss / X_val.shape[0], val_corr / X_val.shape[0]))
        
    def fit(self, sess, x, y, epoches=20, batch_size=32, split=0.2, dropout=1):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=split, random_state=42)
        for epoch in range(epoches):
            print("\nEpoch {:} out of {:}".format(epoch + 1, epoches))
            self.run_epoch(sess, batch_size, (x_train, y_train), (x_valid, y_valid), dropout)
    
    def predict(self, sess, x, batch_size=1024):
        predicts = []
        with tqdm(get_minibatches(x, None, batch_size, False)) as pbar:
            for _x in pbar:
                logits = sess.run(self.outputs, feed_dict={self.images: _x})
                predicts.append(logits)
        return np.concatenate(predicts, axis=0)
     
class Vgg16V2(object):
    def __init__(self, img_rows, img_cols, color_type, learning_rate=0.001, data_format='channels_last'):
        self.data_format = data_format

        self.images = tf.placeholder(tf.float32, [None, img_rows, img_cols, 3], name='images')
        self.labels =  tf.placeholder(tf.uint8, [None, 10], name='labels')

        self.vgg16 = Vgg16('data/vgg16.npy')
        self.vgg16.build(self.images)

        self.flatten = tf.layers.flatten(self.vgg16.pool5)
        self.fc1 = tf.layers.dense(self.flatten, 4096, tf.nn.relu)
        self.fc2 = tf.layers.dense(self.fc1, 4096, tf.nn.relu)
        self.outputs = tf.layers.dense(self.fc2, 10, activation=tf.nn.softmax)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, logits=self.outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def fit(self, sess, x, y, batch_size=1024, nb_epoch=20):
        it = 0
        for _x, _y in BatchIterator((x, y), batch_size=batch_size, epoch=nb_epoch):
            it += 1
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.images: _x, self.labels: _y})
            if it < 11:
                print('Iter {} loss: {}'.format(it, loss))
            if it % 100 == 0:
                print('Iter {} loss: {}'.format(it, loss))
    
    def predict(self, sess, x, batch_size=1024):
        predicts = []
        for _x in BatchIterator(x, batch_size=batch_size, mode='test'):
            logits = sess.run(self.outputs, feed_dict={self.images: _x})
            predicts.append(logits)
        return np.concatenate(predicts, axis=0)
     

def run_transfer_single():
    #input image dimensions
    img_rows, img_cols = 224, 224
    # batch_size = 128
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

    '''
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    '''

    with tf.Graph().as_default():
        vgg16v1 = Vgg16V1(img_rows, img_cols, color_type, data_format='channels_first', learning_rate=0.001)
        #vgg16v1 = Vgg16V2(img_rows, img_cols, color_type, data_format='channels_first')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            sess.run(init, options=run_options)
            vgg16v1.fit(sess, train_data, train_target, batch_size=batch_size)
            print("fit done....................")
            save_path = saver.save(sess, 'model/vgg16v1_model_{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

def load_test_file_names():
    path = os.path.join('..', 'input', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    return files

def chunks(lst, n):
    n = max(1, n)
    return (lst[i:i+n] for i in range(0, len(lst), n))

def run_transfer_predict(model_name):
    img_rows, img_cols = 224, 224
    color_type = 3

    tf.reset_default_graph()

    files = load_test_file_names()
    total_size = len(files)
    file_chunks = chunks(files, total_size//10)

    vgg16v1 = Vgg16V1(img_rows, img_cols, color_type, data_format='channels_first', learning_rate=0.001)
    saver = tf.train.Saver()
    
    count = 0
    predictions = []
    test_id = []
    with tf.Session() as sess:
        saver.restore(sess, 'model/{}.ckpt'.format(model_name))
        for chunk in file_chunks:
            X_test = []
            X_test_id = []
            for fl in chunk:
                flbase = os.path.basename(fl)
                img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
                count += 1
                X_test.append(img)
                X_test_id.append(flbase)
            test_id.append(X_test_id)
            X_test = np.array(X_test)
            # X_test = X_test.transpose((0,3,1,2))
            print('Read {} images from {}'.format(count, total_size))
            print('X_test.shape', X_test.shape)
            predictions.append(vgg16v1.predict(sess, X_test, batch_size=8))
    predictions = np.concatenate(predictions, axis=0)
    test_id = reduce(lambda x,y: x+y, test_id)
    save_submission(test_id, predictions)

if __name__ == '__main__':
    # run_transfer_single()
    run_transfer_predict('vgg16v1_model_2018-03-23_13:05:22')
