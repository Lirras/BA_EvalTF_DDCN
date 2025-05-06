
import numpy as np
import tensorflow
import keras
import seaborn as sns
import keras_data_loader
import copied_models
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    lr_schedule, optimizer = lr_optim()

    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    # model = test_model()
    model = keras.Sequential()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=6, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    freezing(model, 0)

    model.layers[1] = keras.layers.BatchNormalization()
    model.layers[2] = keras.layers.Flatten()
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=4, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    freezing(model, 1)
    model.summary()
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


def lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03, amsgrad=True)
    return lr_schedule, optimizer


def cascade_network():
    lb = LabelBinarizer()

    clear()

    a, b, c, d = keras_data_loader.mnist_loader()

    lr, optim = lr_optim()
    # model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model = copied_models.mnist_model()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.fit(a, b, epochs=10, batch_size=128, validation_data=(c, d), callbacks=[lr])
    model.summary()
    # model = test_model()


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False
    # for layer in model.layers:
    #     layer.trainable = False


def predict(model, train, val):
    model.fit(train, batch_size=None, epochs=10, validation_data=val)
    freezing(model, 0)


schedule()
# cascade_network()
