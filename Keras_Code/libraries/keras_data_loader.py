import numpy as np
import keras
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_digits

NORM = False


def svhn_loader(percentage):

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

    # Very Less: 1%; Less: 30%; Middle: 50%; Much: 70%; Very Much: 99% of Trainingsdata
    X_train, xval, y_train, yval = train_test_split(train_img, train_labels, test_size=percentage, random_state=22)  # 0.99/0.3
    x, X_val, y, y_val = train_test_split(train_img, train_labels, test_size=0.01, random_state=22)  # 0.01/0.7

    # test Dataset reduction
    # test_img, bin_one, test_labels, bin_two = train_test_split(test_img, test_labels, test_size=0.99, random_state=22)  # 0.99
    # print(X_train.shape)
    # print(y_train.shape)

    test_img = test_img.astype('float64')
    test_labels = test_labels.astype('int64')

    # less Data
    train_img = X_train.astype('float64')
    train_labels = y_train.astype('int64')
    val_img = X_val.astype('float64')
    val_labels = y_val.astype('int64')

    # Full Data
    # train_img = train_img.astype('float64')
    # test_img = test_img.astype('float64')

    # train_labels = train_labels.astype('int64')
    # test_labels = test_labels.astype('int64')

    if NORM == True:
        train_img = train_img / 255
        val_img = val_img / 255
        test_img = test_img / 255

    return train_img, train_labels, val_img, val_labels, test_img, test_labels


def mnist_loader():
    (train_dat, train_lb), (val_dat, val_lb) = keras.datasets.mnist.load_data()

    # Upscaling for compatibility with svhn
    train_dat = np.pad(train_dat, ((0, 0), (0, 4), (0, 4)), 'constant')
    val_dat = np.pad(val_dat, ((0, 0), (0, 4), (0, 4)), 'constant')
    val_dat, test_dat, val_lb, test_lb = train_test_split(val_dat, val_lb, test_size=0.5, random_state=22)

    # Scale images to the [0, 1] range
    if NORM is True:
        train_dat = train_dat.astype("float32") / 255
        val_dat = val_dat.astype("float32") / 255
        test_dat = test_dat.astype("float32") / 255
    else:
        train_dat = train_dat.astype("float32")
        val_dat = val_dat.astype("float32")
        test_dat = test_dat.astype("float32")

    # Make sure images have shape (32, 32, 1)
    train_dat = np.expand_dims(train_dat, -1)
    val_dat = np.expand_dims(val_dat, -1)
    test_dat = np.expand_dims(test_dat, -1)

    reduction = False

    if reduction is True:
        train_dat, bin1, train_lb, binlb1 = train_test_split(train_dat, train_lb, test_size=0.99, random_state=22)
        val_dat, bin2, val_lb, binlb2 = train_test_split(val_dat, val_lb, test_size=0.99, random_state=22)

    # 5 K-Fold, dann mehrmals laufen lassen (mit unterschiedlichem State) und das dann zusammen plotten (mean, std)
    # sklearn.Kfold(Train, Test) und dann iterieren
    # Alles mal in einem Plot speichern

    print("train_dat shape:", train_dat.shape)
    print(train_dat.shape[0], "train samples")
    print(train_lb.shape[0], "label samples")

    lb = LabelBinarizer()
    train_lb = lb.fit_transform(train_lb)
    val_lb = lb.fit_transform(val_lb)
    test_lb = lb.fit_transform(train_lb)

    test_lb = test_lb.astype('int64')
    val_lb = val_lb.astype('int64')
    train_lb = train_lb.astype('int64')

    return train_dat, train_lb, val_dat, val_lb, test_dat, test_lb


def load():
    train = loadmat("D:/SVHN/train_32x32.mat")
    test = loadmat("D:/SVHN/test_32x32.mat")
    return train, test


def boston_loader():
    (x_train, x_test), (y_train, y_test) = keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    x_tr, y_tr, x_ts, y_ts = train_test_split(x_train, x_test, train_size=0.5, random_state=42)

    x_tr = boston_delete_columns(x_tr)
    x_tr = boston_change_house_age(x_tr)
    y_tr = boston_delete_columns(y_tr)
    y_tr = boston_change_house_age(y_tr)
    # x_train = boston_delete_columns(x_train)
    # x_train = boston_change_house_age(x_train)
    y_train = boston_delete_columns(y_train)
    y_train = boston_change_house_age(y_train)

    print(x_tr.shape)  # 80, 3
    print(y_tr.shape)  # 324, 3
    print(y_train.shape)  # 102, 3

    if NORM == True:
        # Normalize Trainings Data:
        '''mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train = norm_regr(x_train, mean, std)
        x_test = norm_regr(x_test, mean, std)'''
        mean = x_tr.mean(axis=0)
        std = x_tr.std(axis=0)
        x_tr = norm_regr(x_tr, mean, std)
        x_ts = norm_regr(x_ts, mean, std)

        mean = y_tr.mean(axis=0)
        std = y_tr.std(axis=0)
        y_tr = norm_regr(y_tr, mean, std)
        y_ts = norm_regr(y_ts, mean, std)

        # Normalize Test Data:
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        y_train = norm_regr(y_train, mean, std)
        y_test = norm_regr(y_test, mean, std)

    '''x_train = x_train - mean
    x_train = x_train / std
    x_test = x_test - mean
    x_train = x_train / std'''

    '''print('Boston Dataset:')
    print(x_train[1], x_test[1])
    print(x_train[5], x_test[5])
    print(x_train[68], x_test[68])
    print(x_train[239], x_test[239])
    print(x_train[395], x_test[395])'''

    return x_tr, x_ts, y_tr, y_ts, y_train, y_test
    # return x_train, x_test, y_train, y_test


def boston_delete_columns(x_train):
    i = 11
    while i >= 0:
        if i != 5 and i != 6:
            x_train = np.delete(x_train, i, 1)
        i -= 1
    return x_train


def boston_change_house_age(x_train):
    x_train = x_train[:, [1, 0, 2]]  # Change Columns
    # Only 85% of House Age. Reason: Houses prior to 1940 are 85 years old or older.
    # Maximum of Houses are 100
    c = 0
    while c < len(x_train):
        x_train[c, 0] = (x_train[c, 0] * 85)/100
        # Make from anti-proportional to proportional
        x_train[c, 2] = 40 - x_train[c, 2]  # Without is better
        c += 1
    return x_train

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
    # --> MEDV Median value of owner - occupied homes in $1000's This one is the actual price


def california_loader():
    (x_train, x_test), (y_train, y_test) = keras.datasets.california_housing.load_data(
        version="small", path="california_housing.npz", test_split=0.2, seed=113
    )

    x_tr, y_tr, x_ts, y_ts = train_test_split(x_train, x_test, test_size=0.5, random_state=42)
    x_tr = california_delete_columns(x_tr)
    x_ts = california_change_prices(x_ts)
    y_tr = california_delete_columns(y_tr)
    y_ts = california_change_prices(y_ts)

    # x_train = california_delete_columns(x_train)
    # x_test = california_change_prices(x_test)
    y_train = california_delete_columns(y_train)
    y_test = california_change_prices(y_test)

    print(x_tr.shape)  # 4, 3
    print(y_tr.shape)  # 476,3
    print(y_train.shape)  # 120, 3

    if NORM == True:
        # Normalize Data:
        '''mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train = norm_regr(x_train, mean, std)
        x_test = norm_regr(x_test, mean, std)'''

        mean = x_tr.mean(axis=0)
        std = x_tr.std(axis=0)
        x_tr = norm_regr(x_tr, mean, std)
        x_ts = norm_regr(x_ts, mean, std)


        mean = y_tr.mean(axis=0)
        std = y_tr.std(axis=0)
        y_tr = norm_regr(y_tr, mean, std)
        y_ts = norm_regr(y_ts, mean, std)

        # Normalize Test Data:
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        y_train = norm_regr(y_train, mean, std)
        y_test = norm_regr(y_test, mean, std)

    '''x_train -= mean
    x_train /= std
    x_test -= mean
    x_train /= std'''

    # longitude, latitude, age, totalRooms, Bedrooms, pop, households, median income

    # Used columns: HouseAge, AveRoomsPerDwelling, MedInc
    '''print('California Dataset')
    print(x_train[1], x_test[1])
    print(x_train[4], x_test[4])
    print(x_train[200], x_test[200])
    print(x_train[1000], x_test[1000])
    print(x_train[8943], x_test[8943])
    print(x_train[5032], x_test[5032])
    print(x_train[12000], x_test[12000])
    print(x_train[14532], x_test[14532])
    print(x_train[3902], x_test[3902])
    print(x_train[6241], x_test[6241])'''

    return x_tr, x_ts, y_tr, y_ts, y_train, y_test
    # return x_train, x_test, y_train, y_test


def california_delete_columns(x_train):

    # total rooms/households = Rooms per Dwelling
    c = 0
    while c < len(x_train):
        x_train[c, 3] = x_train[c, 3] / x_train[c, 6]
        c += 1

    # Delete unused columns:
    i = 6
    while i >= 0:
        if i != 2 and i != 3:
            x_train = np.delete(x_train, i, 1)
        i -= 1

    return x_train


def california_change_prices(x_test):

    c = 0
    while c < len(x_test):
        x_test[c] = x_test[c]/1000
        c += 1
    return x_test

    # MedInc: median income in block group
    # HouseAge: median house age in block group
    # AveRooms: average number of rooms per household
    # AveBedrms: average number of bedrooms per household
    # Population: block group population
    # AveOccup: average number of household members
    # Latitude: block group latitude
    # Longitude: block group longitude


def norm_regr(arr, mean, std):
    ls = []
    for item in arr:
        ls.append((item - mean)/std)
    return np.array(ls)


def digit_loader():
    # N, H*W, None -> 1797, 64 (16x Input Size -> 16zeros!)
    digits = load_digits()
    data = np.array(digits['data'])
    label = np.array(digits['target'])
    print(data.shape)
    print(label.shape)


# svhn_loader()
