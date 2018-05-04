#inspired by https://www.kaggle.com/zfturbo/keras-sample

from data_preprocess import *
from naive_model import NaiveModel
from batch_iterator import BatchIterator
import logging

def run_single():
    #input image dimensions
    img_rows, img_cols = 64, 64
    batch_size = 32
    nb_epoch = 1
    random_state = 51

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type=1)
    logging.info('train_data shape:', train_data.shape)
    logging.info('train_target shape:', train_target.shape)
    logging.info('dirver_id shape:', len(driver_id))
    logging.info('unique_drivers shape:', len(unique_drivers))

    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
    # unique_list_train = ['p002', 'p012', 'p014']
    X_train, Y_train = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    logging.info('X_train shape:', X_train.shape)
    logging.info('Y_train shape:', Y_train.shape)
    unique_list_valid = ['p081']
    X_valid, Y_valid = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)
    logging.info('X_valid shape:', X_valid.shape)
    logging.info('Y_valid shape:', Y_valid.shape)

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
    logging.info('test_data shape:', test_data.shape)
    logging.info('test_id shape:', len(test_id))

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
    run_prediction("navie_model_2018-03-07_18:03:29")
