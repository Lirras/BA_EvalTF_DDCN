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


def schedule():
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

    '''datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,
        zoom_range=[0.95, 1.05],
        height_shift_range=0.10,
        shear_range=0.15
    )'''

    keras.backend.clear_session()

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

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(lr=0.0001, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(X_train, y_train, batch_size=128,
                                  epochs=1, validation_data=(X_val, y_val),
                                  callbacks=[lr_schedule])

    # todo: Build model for this
    # model.add(keras.layers.BatchNormalization())


def load():
    train = loadmat("D:/SVHN/train_32x32.mat")
    test = loadmat("D:/SVHN/test_32x32.mat")
    return train, test


schedule()
