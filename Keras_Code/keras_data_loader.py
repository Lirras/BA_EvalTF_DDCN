import numpy as np
import keras
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer


def svhn_loader():

    train, test = load()
    train_img = np.array(train['X'])
    test_img = np.array(test['X'])

    train_labels = train['y']
    test_labels = test['y']

    print(train_img.shape)  # 32 32 3 73257
    print(test_img.shape)  # 32 32 3 26032

    train_img = np.moveaxis(train_img, -1, 0)  # B, H, W, C
    test_img = np.moveaxis(test_img, -1, 0)  # B, H, W, C

    # GrayScaling for compatibility with mnist
    train_img = np.dot(train_img[..., :3], [0.2989, 0.5870, 0.1140])
    test_img = np.dot(test_img[..., :3], [0.2989, 0.5870, 0.1140])

    print(train_img.shape)
    print(test_img.shape)

    print('Min: {}, Max: {}'.format(train_img.min(), train_img.max()))

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    # X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.15, random_state=22)

    train_img = train_img.astype('float64')
    test_img = test_img.astype('float64')

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    return train_img, train_labels, test_img, test_labels


def mnist_loader():
    (train_dat, train_lb), (val_dat, val_lb) = keras.datasets.mnist.load_data()

    # Upscaling for compatibility with svhn
    train_dat = np.pad(train_dat, ((0, 0), (0, 4), (0, 4)), 'constant')
    val_dat = np.pad(val_dat, ((0, 0), (0, 4), (0, 4)), 'constant')

    # Scale images to the [0, 1] range
    train_dat = train_dat.astype("float32") / 255
    val_dat = val_dat.astype("float32") / 255
    # Make sure images have shape (32, 32, 1)
    train_dat = np.expand_dims(train_dat, -1)
    val_dat = np.expand_dims(val_dat, -1)

    print("train_dat shape:", train_dat.shape)
    print(train_dat.shape[0], "train samples")
    print(train_lb.shape[0], "label samples")

    lb = LabelBinarizer()
    train_lb = lb.fit_transform(train_lb)
    val_lb = lb.fit_transform(val_lb)

    val_lb = val_lb.astype('int64')
    train_lb = train_lb.astype('int64')

    return train_dat, train_lb, val_dat, val_lb


def load():
    train = loadmat("D:/SVHN/train_32x32.mat")
    test = loadmat("D:/SVHN/test_32x32.mat")
    return train, test
