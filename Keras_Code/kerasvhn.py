
# import numpy as np
import tensorflow
import keras
# import seaborn as sns
import keras_data_loader
import copied_models
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    lr_schedule, optimizer = lr_optim()

    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    # model = test_model()
    model = keras.Sequential()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
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


def lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03, amsgrad=True)
    return lr_schedule, optimizer


def cascade_network():

    clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.mnist_model()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    predict_train(model, a, b, c, d, lr, 0)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, a, b, c, d, lr, 1)

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    predict_train(model, a, b, c, d, lr, 2)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, a, b, c, d, lr, 3)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, 'softmax'))
    model.fit(e, f, batch_size=128, epochs=4, validation_data=(g, h), callbacks=[lr])
    # 1 epoch: ACC:20%; ValACC:50%
    # 4 epoch: ACC:76%; ValACC:73%

    model.summary()


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False
    # for layer in model.layers:
    #     layer.trainable = False


def predict_train(model, train_dat, train_lb, val_dat, val_lb, lr, freeze):

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.fit(train_dat, train_lb, batch_size=128, epochs=1, validation_data=(val_dat, val_lb), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    model.pop()


# schedule()
cascade_network()
