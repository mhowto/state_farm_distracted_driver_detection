from starter import *

import os, time, glob
from functools import reduce

import numpy as np
from tqdm import tqdm
import pickle

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
    start_time = time.time()
    driver_id = []
    X_train_group = None
    y_train_group = None
    X_train_id_group = []

    print('Read train images')
    for j in range(10):
        X_train, y_train, X_train_id = [], [], []
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
        cache_train([X_train, y_train, X_train_id], img_rows, img_cols, color_type, j)
        # X_train_group.append(X_train)
        # y_train_group.append(y_train)
        X_train_id_group.append(X_train_id)
        driver_id.append(X_train_id)

    driver_id = reduce(lambda h,t: h+t, driver_id)
    unique_drivers = sorted(list(set(driver_id)))
    path = os.path.join('cache', unique_drivers_file)
    with open(path, 'wb') as f:
        pickle.dump(unique_drivers, f)
    with open(os.path.join('cache', cache_train_flag), 'w') as f:
        f.write('1')
    # X_train_group = np.concatenate(X_train_group, axis=0)
    # y_train_group = np.concatenate(y_train_group, axis=0)
    X_train_id_group = reduce(lambda x,y:x+y, X_train_id_group)
    
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train_group, y_train_group, X_train_id_group, unique_drivers

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
        img2driver = get_driver_data()
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
        img2driver = get_driver_data()
    else:
        print('Restore train from cache!')
        test_data, test_id = restore_test(img_rows, img_cols, color_type)
    
    return test_data, test_id

def run_transfer_single():
    #input image dimensions
    img_rows, img_cols = 244, 244
    batch_size = 1024
    nb_epoch = 20
    random_state = 51
    color_type = 3

    '''
    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data2(img_rows, img_cols, color_type=color_type)
    print('train_data shape:', train_data.shape)
    print('train_target shape:', train_target.shape)
    print('dirver_id shape:', len(driver_id))
    print('unique_drivers shape:', len(unique_drivers))
    '''

    test_data, test_id = read_and_normalize_test_data2(img_rows, img_cols, color_type=color_type)
    print('test_data shape:', test_data.shape)
    print('test_id shape:', test_id.shape)


if __name__ == '__main__':
    run_transfer_single()