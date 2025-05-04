# Copied from https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc

import numpy as np
import tensorflow
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
# from keras.preprocessing.image import ImageDataGenerator
# matplotlib inline
import data_loader
import torch


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    lr_schedule, optimizer = lr_optim()

    train_data, train_label, test_data, test_label = preparation()

    model = test_model()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=30, validation_data=(test_data, test_label), callbacks=[lr_schedule])


    # label_data = np.array(svhn_train['y'])

    # for freezing weights
    '''for layer in model.layers[:5]:  # Bis Layer 5 gefreezt, rest ist normal berechnet.
        layer.trainable = False
    for layer in model.layers[5:]:
        layer.trainable = True'''

    # svhn_train = np.moveaxis(svhn_train.dataset, -1, 0)
    # svhn_val = np.moveaxis(svhn_val.dataset, -1, 0)

    '''batch_size = 128

    model = keras.Sequential([])
    svhn_train, svhn_val = data_loader.svhn_data_loader(batch_size)

    lr_schedule, optimizer = lr_optim()

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, 32, 32)))  # 32 32 1


    # Configures the model for training
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # for data, label in svhn_train:
    history = model.fit(svhn_train, batch_size=None, epochs=1, validation_data=svhn_val, callbacks=[lr_schedule])'''
    # Dataloader Object possible for model.fit as x; y should be empty then.

    # todo: Build model for this
    # model.add(keras.layers.BatchNormalization())


def load():
    train = loadmat("D:/SVHN/train_32x32.mat")
    test = loadmat("D:/SVHN/test_32x32.mat")
    return train, test


def lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03, amsgrad=True)
    return lr_schedule, optimizer


def preparation():
    '''datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=8,
            zoom_range=[0.95, 1.05],
            height_shift_range=0.10,
            shear_range=0.15
        )'''

    train, test = load()
    train_img = np.array(train['X'])
    test_img = np.array(test['X'])

    train_labels = train['y']
    test_labels = test['y']

    print(train_img.shape)  # 32 32 3 73257
    print(test_img.shape)  # 32 32 3 26032

    train_img = np.moveaxis(train_img, -1, 0)
    test_img = np.moveaxis(test_img, -1, 0)

    print(train_img.shape)
    print(test_img.shape)

    print('Min: {}, Max: {}'.format(train_img.min(), train_img.max()))

    # train_img /= 255.0
    # test_img /= 255.0

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.15, random_state=22)
    print(y_val.shape)
    # plt.imshow(train_img[13529])
    # plt.show()
    # print('Label: ', train_labels[13529])

    train_img = train_img.astype('float64')
    test_img = test_img.astype('float64')

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    return train_img, train_labels, test_img, test_labels


def test_model():
    model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation='relu',
                                input_shape=(32, 32, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation='softmax')
        ])
    return model


def cascade_network():
    # lb = LabelBinarizer()

    clear()
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    svhn_train, svhn_val = data_loader.svhn_data_loader(128)

    # mnist_train_data = torch.permute(mnist_train.dataset[0], (0, 2, 3, 1))
    print(mnist_train.dataset[0][0].shape)  # Input
    print(mnist_train.dataset[0][1])  # Target
    print(mnist_train.dataset[0][0].numpy())  # To Numpy

    # todo: It must exist a better way!
    train_data_input = []  # np.array([])
    train_data_target = []  # np.array([])
    for data, label in mnist_train:
        train_data_input.append(data.numpy())
        train_data_target.append(label.numpy())

    train_data_target = np.array(train_data_target)
    train_data_input = np.array(train_data_input)

    valid_data_input = []
    valid_data_target = []
    for data, label in mnist_val:
        valid_data_input.append(data.numpy())
        valid_data_target.append(label.numpy())

    valid_data_target = np.array(valid_data_target)
    valid_data_input = np.array(valid_data_input)

    # print(train_data_input)
    # print(train_data_target)
    # torch.permute(mnist_val.dataset['data'], (0, 2, 3, 1))

    '''mnist_train = lb.fit_transform(mnist_train)
    mnist_val = lb.fit_transform(mnist_val)
    svhn_train = lb.fit_transform(svhn_train)
    svhn_val = lb.fit_transform(svhn_val)'''

    lr, optim = lr_optim()
    model = keras.Sequential()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.fit(train_data_input, train_data_target, batch_size=None, epochs=10, validation_data=(valid_data_input, valid_data_target))

    return
    # predict(model, mnist_train, mnist_val)

    model.add(keras.layers.Reshape((1, 32, 32)))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    predict(model, svhn_train, svhn_val)

    model.add(keras.layers.BatchNormalization())
    predict(model, svhn_train, svhn_val)


def clear():
    keras.backend.clear_session()


def freezing(model):
    for layer in model.layers:
        layer.trainable = False


def predict(model, train, val):
    model.fit(train, batch_size=None, epochs=10, validation_data=val)
    freezing(model)


# schedule()
cascade_network()
