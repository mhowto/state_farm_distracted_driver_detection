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
from keras.utils import to_categorical

DEFAULT_DRIVER_FILE = os.path.join('data', 'driver_imgs_list.csv')
DEFAULT_TRAIN_DATA_PATH = os.path.join('data', 'train')
DEFAULT_TEST_DATA_PATH = os.path.join('data', 'test')

use_cache = 1

import sys

def get_im_cv2_mod(filename, img_rows, img_cols, color_type=1):
    """Returns resized image matrixes.

    Arguments:
        color_type: 1 - gray; 3 - RGB.
    """
    if color_type == 1:
        img = cv2.imread(filename, 0)
    else:
        img = cv2.imread(filename)
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    return resized

def get_driver_data(filename=DEFAULT_DRIVER_FILE):
    img2driver = dict()
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

    for j in range(10):
        file_pattern = os.path.join(path, 'c'+str(j), '*.jpg')
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
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    print('cache_path:', cache_path)
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore train from cache!')
        test_data, test_id = restore_data(cache_path)
    
    test_data = np.array(test_data, dtype='f')
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data /= 255
    return test_data, test_id

def read_and_normalize_test_data2(img_rows, img_cols, path, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_' + path + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type, path)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore train from cache!')
        test_data, test_id = restore_data(cache_path)
    
    test_data = np.array(test_data, dtype='f')
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data /= 255
    return test_data, test_id

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    selected_index = []
    for i, driver in enumerate(driver_id):
        if driver in driver_list:
            selected_index.append(i)
    return train_data[selected_index, ...], train_target[selected_index, ...]

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

def split_partial_train_data(path, target_path, ration=0.2):
    for c in os.listdir(path):
        files =os.listfiles(os.path.join(path, c))
