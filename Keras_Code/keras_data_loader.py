import numpy as np
import keras
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_digits


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

    train_img = np.expand_dims(train_img, -1)
    test_img = np.expand_dims(test_img, -1)

    print(train_img.shape)
    print(test_img.shape)

    print('Min: {}, Max: {}'.format(train_img.min(), train_img.max()))

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.99, random_state=22)
    # print(X_train.shape)
    # print(y_train.shape)

    train_img = X_train.astype('float64')
    train_labels = y_train.astype('int64')

    # train_img = train_img.astype('float64')
    test_img = test_img.astype('float64')

    # train_labels = train_labels.astype('int64')
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


def boston_loader():
    (x_train, x_test), (y_train, y_test) = keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )
    print(x_train.shape)  # 404 13

    # --> CRIM per capita crime rate by town
    # --> ZN proportion of residential land zoned for lots over 25, 000 sq.ft.
    # --> INDUS proportion of non - retail business acres per town
    # --> CHAS Charles River dummy variable( = 1 if tract bounds river; 0 otherwise)
    # --> NOX nitric oxides concentration(parts per 10 million)
    # RM average number of rooms per dwelling - AveRooms
    # AGE proportion of owner - occupied units built prior to 1940 - HouseAge
    # --> DIS weighted distances to five Boston employment centres
    # --> RAD index of accessibility to radial highways
    # --> TAX full - value property - tax rate per $10, 000
    # --> PTRATIO pupil - teacher ratio by town
    # --> B 1000(Bk - 0.63) ^ 2 where Bk is the proportion of blacks by town
    # LSTAT % lower status of the population - MedInc
    # --> MEDV Median value of owner - occupied homes in $1000's


def california_loader():
    (x_train, x_test), (y_train, y_test) = keras.datasets.california_housing.load_data(
        version="large", path="california_housing.npz", test_split=0.2, seed=113
    )
    print(x_train.shape)  # 16512 8

    # MedInc: median income in block group
    # HouseAge: median house age in block group
    # AveRooms: average number of rooms per household
    # AveBedrms: average number of bedrooms per household
    # Population: block group population
    # AveOccup: average number of household members
    # Latitude: block group latitude
    # Longitude: block group longitude


def digit_loader():
    # N, H*W, None -> 1797, 64 (16x Input Size -> 16zeros!)
    digits = load_digits()
    data = np.array(digits['data'])
    label = np.array(digits['target'])
    print(data.shape)
    print(label.shape)
