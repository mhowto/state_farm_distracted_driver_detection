#inspired by https://www.kaggle.com/zfturbo/keras-sample

import os
import time
import csv
import random
import glob
import pickle
from datetime import datetime

import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.utils import to_categorical
import tensorflow as tf

from naive_model import NaiveModel
from batch_iterator import BatchIterator

DEFAULT_DRIVER_FILE = os.path.join('data', 'driver_imgs_list.csv')
DEFAULT_TRAIN_DATA_PATH = os.path.join('data', 'train')
DEFAULT_TEST_DATA_PATH = os.path.join('data', 'test')

use_cache = 1

import sys

def get_im_cv2_mod(filename, img_rows, img_cols, color_type=1):
    """Returns rotated and resized image matrixes.

    Arguments:
        color_type: 1 - gray; 3 - RGB.
    """
    if color_type == 1:
        img = cv2.imread(filename, 0)
    else:
        img = cv2.imread(filename)
    # Reduce size
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    # return resized.astype(np.float16)
    # print('resized dtype:', resized.dtype)
    # print('resized shape:', resized.shape)
    # print('resized size:', resized.nbytes)
    return resized

def get_driver_data(filename=DEFAULT_DRIVER_FILE):
    img2driver = dict()
    print('Read drivers data')
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            img2driver[line['img']] = line['subject']

    return img2driver

def plot_img_cv2mat(img, color=1):
    """ Plot cv2 image.

    Arguments:
        color: 1-gray, 3-rgb.
    """
    if color == 1:
        cmap = 'gray'
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None

    plt.imshow(img, cmap=cmap, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def plot_img_file(filename, color=1):
    """ Plot image.
    Arguments:
        color: 1-gray, 3-rgb.
    """
    if color == 1:
        img = cv2.imread(filename, 0)
    else:
        img = cv2.imread(filename, 3)
    plot_img_cv2mat(img, color)

def load_train(img_rows, img_cols, color_type=1, path=DEFAULT_TRAIN_DATA_PATH):
    X_train = []
    y_train = []
    X_train_id = []
    start_time = time.time()

    print('Read train images')
    for j in range(10):
        file_pattern = os.path.join(path, 'c'+str(j), '*.jpg')
        print('Load folder {}'.format(file_pattern))
        for fl in tqdm(glob.glob(file_pattern)):
            X_train_id.append(os.path.basename(fl))
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
    
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def load_test(img_rows, img_cols, color_type=1, path=DEFAULT_TEST_DATA_PATH):
    X_test, X_test_id = [], []
    start_time = time.time()
    print('Read test images')
    file_pattern = os.path.join(path, '*.jpg')
    for fl in tqdm(glob.glob(file_pattern)):
        X_test_id.append(os.path.basename(fl))
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img)

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        print('Directory doesn\'t exists')

def restore_data(path):
    data = {}
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_and_normalize_train_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id = load_train(img_rows, img_cols, color_type)
        img2driver = get_driver_data()

        driver_id = []
        for idx in train_id:
            driver_id.append(img2driver[idx])
        unique_drivers = sorted(list(set(driver_id)))
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        train_data, train_target, driver_id, unique_drivers = restore_data(cache_path)
    
    train_data = np.array(train_data, dtype='f')
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = np.reshape(train_data, (train_data.shape[0], color_type, img_rows, img_cols))
    train_target = to_categorical(train_target)
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train_samples')
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore train from cache!')
        test_data, test_id = restore_data(cache_path)
    
    test_data = np.array(test_data, dtype='f')
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    selected_index = []
    for i, driver in enumerate(driver_id):
        if driver in driver_list:
            selected_index.append(i)
    return train_data[selected_index, ...], train_target[selected_index, ...]

def run_single():
    #input image dimensions
    img_rows, img_cols = 64, 64
    batch_size = 32
    nb_epoch = 1
    random_state = 51

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type=1)
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

    with tf.Graph().as_default():
        nn = NaiveModel(img_rows, img_cols, 10, color_type=1)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # nn.fit(sess, X_train, Y_train, nb_epoch=400)
            nn.fit(sess, X_train, Y_train)
            predictions_valid = nn.predict(sess, X_valid)
            loss = log_loss(Y_valid, predictions_valid)
            print('validate loss: {}'.format(loss))
            save_path = saver.save(sess, 'model/navie_model_{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

def save_submission(image_ids, predictions):
    with open('result/submission_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
        for i, idx in enumerate(image_ids):
            if i > predictions.shape[0]:
                break
            pred = predictions[i,...]
            row = [idx] + pred.tolist()
            writer.writerow(row)

def run_prediction(model_name):
    tf.reset_default_graph()

    img_rows, img_cols = 64, 64

    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type=1)
    print('test_data shape:', test_data.shape)
    print('test_id shape:', len(test_id))

    nn = NaiveModel(img_rows, img_cols, 10, color_type=1)
    saver = tf.train.Saver()

    predictions = []
    test_data_iter = BatchIterator(test_data, batch_size=1024, mode='test')

    with tf.Session() as sess:
        saver.restore(sess, 'model/{}.ckpt'.format(model_name))
        with tqdm(total=test_data_iter.iters) as pbar:
            for _x in test_data_iter:
                pbar.update(1)
                predict = sess.run(nn.outputs, feed_dict={nn.images: _x, nn.prob: 1.0})
                predictions.append(predict)
            predictions = np.concatenate(predictions, axis=0)
    save_submission(test_id, predictions)

if __name__ == '__main__': 
    '''
    import matplotlib.pyplot as plt
    filename = 'data/train/c0/img_1215.jpg'
    raw_img = cv2.imread(filename)
    plot_img_cv2mat(raw_img, 3)

    img = get_im_cv2_mod(filename, 100, 80, color_type=3)
    plot_img_cv2mat(img, 3)
    '''
    # run_single()
    run_prediction("navie_model_2018-03-07_18:03:29")
