from starter import get_im_cv2_mod, get_driver_data, DEFAULT_TRAIN_DATA_PATH, DEFAULT_TEST_DATA_PATH

import os, time, glob

import numpy as np
import tqdm
import pickle

use_cache = 1 

train_x_pattern = 'train_r_{}_c_{}_t_{}_class_{}_train_x'
train_y_pattern = 'train_r_{}_c_{}_t_{}_class_{}_train_y'
train_id_pattern = 'train_r_{}_c_{}_t_{}_class_{}_train_id.dat'

def load_and_cache_train(img_rows, img_cols, color_type=1, path=DEFAULT_TRAIN_DATA_PATH):
    start_time = time.time()

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
        cache_train([X_train, y_train, X_train_id], img_rows, img_cols, col_type, j)
        # np.save(os.path.join('cache', train_x_pattern.format(img_rows, img_cols, color_type, j)), np.array(X_train))
        # np.save(os.path.join('cache', train_y_pattern.format(img_rows, img_cols, color_type, j)), np.array(y_train))
        # pickle.dump(os.path.join('cache', train_id_pattern.format(img_rows, img_cols, color_type, j)), X_train_id)
    
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def cache_train(raw_data, img_rows, img_cols, color_type, cl):
    X_train, y_train, X_train_id = raw_data
    np.save(os.path.join('cache', train_x_pattern.format(img_rows, img_cols, color_type, cl)), np.array(X_train))
    np.save(os.path.join('cache', train_y_pattern.format(img_rows, img_cols, color_type, cl)), np.array(y_train))
    pickle.dump(os.path.join('cache', train_id_pattern.format(img_rows, img_cols, color_type, cl)), X_train_id)

def restore_train(img_rows, img_cols, color_type, cl):
    X_train = np.load(os.path.join('cache', train_x_pattern.format(img_rows, img_cols, color_type, cl)+'.npy'))
    y_train = np.load(os.path.join('cache', train_y_pattern.format(img_rows, img_cols, color_type, cl)+'.npy'))
    X_train_id = pickle.load(os.path.join('cache', train_id_pattern.format(img_rows, img_cols, color_type, cl)))
    return (X_train, y_train, X_train_id)

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
    return X_test, X_test_id

def read_and_normalize_train_data2(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id = load_and_cache_train(img_rows, img_cols, color_type)
        img2driver = get_driver_data()

        driver_id = []
        for idx in train_id:
            driver_id.append(img2driver[idx])
        unique_drivers = sorted(list(set(driver_id)))
        # pickle.dump()
        # cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        # train_data, train_target, driver_id = restore_train(img_rows, img_cols, color_type, )
        # , unique_drivers = restore_data(cache_path)
    
    train_data = np.array(train_data, dtype='f')
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = np.reshape(train_data, (train_data.shape[0], color_type, img_rows, img_cols))
    train_target = to_categorical(train_target)
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train_samples')
    return train_data, train_target, driver_id, unique_drivers



def run_single():
    #input image dimensions
    img_rows, img_cols = 244, 244
    batch_size = 1024
    nb_epoch = 20
    random_state = 51
    color_type = 3

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type=color_type)
    print('train_data shape:', train_data.shape)
    print('train_target shape:', train_target.shape)
    print('dirver_id shape:', len(driver_id))
    print('unique_drivers shape:', len(unique_drivers))

if __name__ == '__main__':
    run_single()