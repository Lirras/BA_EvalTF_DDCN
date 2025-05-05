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

    # model = test_model()
    model = keras.Sequential()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=6, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    freezing(model, 0)

    model.layers[1] = keras.layers.BatchNormalization()
    model.layers[2] = keras.layers.Flatten()
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=4, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    freezing(model, 1)
    # keras.layers.RandomGrayscale(1.0, 'channels_first')

    '''model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.Dropout(0.3))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.BatchNormalization())
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Dropout(0.3))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.BatchNormalization())
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Dropout(0.3))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Flatten())
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Dropout(0.4))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_data=(test_data, test_label),
              callbacks=[lr_schedule])
    freezing(model)'''


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

    train_img = np.moveaxis(train_img, -1, 0)  # B, H, W, C
    test_img = np.moveaxis(test_img, -1, 0)  # B, H, W, C

    print(train_img.shape)
    print(test_img.shape)

    print('Min: {}, Max: {}'.format(train_img.min(), train_img.max()))

    # train_img /= 255.0
    # test_img /= 255.0
    # print(train_labels[1])

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.15, random_state=22)

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
    lb = LabelBinarizer()

    clear()
    a, b, c, d, e, f, g, h = loader_crutch()  # Batch_size at data_loader is the size of the corresponded dataset
    b = lb.fit_transform(b)
    d = lb.fit_transform(d)
    f = lb.fit_transform(f)
    h = lb.fit_transform(h)
    b = b.astype('int64')
    d = d.astype('int64')
    f = f.astype('int64')
    h = h.astype('int64')
    a = a.astype('float')
    c = c.astype('float')
    e = e.astype('float')
    g = g.astype('float')

    a = np.moveaxis(a, 1, 3)  # B, C, H, W -> B, H, W ,C
    c = np.moveaxis(c, 1, 3)
    e = np.moveaxis(e, 1, 3)
    g = np.moveaxis(g, 1, 3)
    print(a.shape)
    print(c.shape)
    print(e.shape)
    print(g.shape)

    # print(b.shape)

    '''print(mnist_train.dataset[0][0].shape)  # Input
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

    train_data_target = lb.fit_transform(train_data_target)
    train_data_input = lb.fit_transform(train_data_input)
    valid_data_input = lb.fit_transform(valid_data_input)
    valid_data_target = lb.fit_transform(valid_data_target)'''

    '''mnist_train = lb.fit_transform(mnist_train)
    mnist_val = lb.fit_transform(mnist_val)
    svhn_train = lb.fit_transform(svhn_train)
    svhn_val = lb.fit_transform(svhn_val)'''

    lr, optim = lr_optim()
    # model = keras.Sequential()
    model = test_model()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    # model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, 32, 32)))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(1024))
    # model.add(keras.layers.Dense(10, activation='softmax'))
    # a,b = mnist_train/ c,d = mnist_val/ e,f = svhn_train/ g,h = svhn_val as data, label
    # model.fit(a, b, epochs=10, batch_size=128, validation_data=(c, d), callbacks=[lr])  # Why at 10%?
    # freezing(model, 0)
    # freezing(model, 1)
    # model.layers[-1] = keras.layers.Reshape((32, 32))
    # model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, epochs=10, batch_size=128, validation_data=(g, h), callbacks=[lr])  # Not Grayscaled: 15% -> The Dataformat is a problem! Not the Grayscaling
    # todo: There is something wrong with the data!
    # model.fit(e, f, epochs=10, batch_size=128, validation_data=(g, h), callbacks=[lr])  # Why at 18%? -> Well, this is normal
    # with old data at 90%; with new data at 19.5%
    # model.fit_generator(mnist_train, mnist_val)


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False
    # for layer in model.layers:
    #     layer.trainable = False


def predict(model, train, val):
    model.fit(train, batch_size=None, epochs=10, validation_data=val)
    freezing(model, 0)


def loader_crutch():
    mnist_tr, mnist_va = data_loader.mnist_data_loader(True, 1)
    svhn_tr, svhn_va = data_loader.svhn_data_loader(1)
    mnist_tr_dat, mnist_tr_lb = loader_crutch_backend(mnist_tr)
    mnist_va_dat, mnist_va_lb = loader_crutch_backend(mnist_va)
    svhn_tr_dat, svhn_tr_lb = loader_crutch_backend(svhn_tr)
    svhn_va_dat, svhn_va_lb = loader_crutch_backend(svhn_va)
    return mnist_tr_dat, mnist_tr_lb, mnist_va_dat, mnist_va_lb, svhn_tr_dat, svhn_tr_lb, svhn_va_dat, svhn_va_lb


def loader_crutch_backend(dataloader):
    data = next(iter(dataloader))[0].numpy()
    label = next(iter(dataloader))[1].numpy()
    return data, label


# schedule()
cascade_network()
