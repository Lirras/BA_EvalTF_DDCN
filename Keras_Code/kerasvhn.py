
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

    m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = keras_data_loader.mnist_loader()
    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
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

    predict_train(model, e, f, g, h, lr, 0)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, 'softmax'))
    model.fit(e, f, batch_size=128, epochs=1, validation_data=(g, h), callbacks=[lr])

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
