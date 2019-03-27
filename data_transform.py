from six.moves import cPickle as pickle
import os
import numpy as np
from keras.utils import np_utils
from tqdm import tqdm
import cv2
import skimage.transform



def load_pickle(f):
    return pickle.load(f, encoding='latin1')


def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = load_pickle(f)
        X = data_dict['data']
        Y = data_dict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)

    return X, Y


def load_cifar10(ROOT):
    xs = []
    ys = []

    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    del X, Y

    Xte, Yte = load_cifar_batch(os.path.join(ROOT, 'test_batch'))

    return (Xtr, Ytr), (Xte, Yte)


def get_cifar10_data(img_rows, img_cols):
    num_classes = 10
    cifar10_dir = '/home/lst/datasets/cifar-10-batches-py/'

    (x_train, y_train), (x_test, y_test) = load_cifar10(cifar10_dir)

    # X_tr = []
    # for i in tqdm(range(len(x_train))):
    #     X_tr.append(skimage.transform.resize(x_train[i], (img_rows, img_cols)))
    #
    # X_te = []
    # for i in range(len(x_test)):
    #     X_te.append(skimage.transform.resize(x_test[i], (img_rows, img_cols)))

    x_train = np.array([skimage.transform.resize(img, (img_rows, img_cols)) for img in tqdm(x_train[:30000, :, :, :])])
    x_test = np.array([skimage.transform.resize(img, (img_rows, img_cols)) for img in x_test[:30000, :, :, :]])
    y_train = y_train[:30000]
    y_test = y_test[:30000]
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # x_train = np.concatenate(X_tr)
    # x_test = np.concatenate(X_te)
    # x_train = x_train.resize((50000, 224, 224, 3))
    # x_test = x_test.resize((50000, 224, 224, 3))
    x_train /= 255.0
    x_test /= 255.0

    return (x_train, y_train), (x_test, y_test)
