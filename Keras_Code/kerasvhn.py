import numpy
import numpy as np
import tensorflow
import keras
import seaborn as sns
import keras_data_loader
import copied_models
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix


def schedule():
    batch_size = 128
    # sns.set_theme()
    keras.backend.clear_session()
    lr_schedule, optimizer = lr_optim()

    m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = keras_data_loader.mnist_loader()
    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    '''tips = sns.load_dataset('tips')
    sns.relplot(
        data=tips,
        x="total_bill", y="tip", col="time",
        hue="smoker", style="smoker", size="size",
    )'''
    # plt.show()  # todo: Need another version of matplotlib!

    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.svhn_model()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    # model.fit(train_data, train_label, batch_size=128, epochs=30, validation_data=(test_data, test_label), callbacks=[lr_schedule])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(m_tr_dat, m_tr_lb, batch_size=128, epochs=1, validation_data=(m_val_dat, m_val_lb), callbacks=[lr_schedule])
    freezing(model, 0)
    freezing(model, 1)
    model.pop()
    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.RNN())
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 2, 1)
    freezing(model, 3)

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=128, epochs=10, validation_data=(test_data, test_label), callbacks=[lr_schedule])

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

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    model.pop()
    model.pop()
    # predict_train(model, e, f, g, h, lr, 0)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 1, 1)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 2, 8)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 3, 8)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 4, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 3, 8)
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 5, 8)
    model.add(keras.layers.MaxPooling2D(4, 4))
    predict_train(model, e, f, g, h, lr, 6, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 7, 1)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    post_flatten(model, e, f, g, h, lr, 8)
    freezing(model, 9)

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=1, validation_data=(g, h), callbacks=[lr])
    # 93.7 TF 11.7 -> 64.4
    # Conv8epochs: 92.7 TF 11.0 -> 68.5

    model.summary()
    # TF: 75.8%
    # ohne: 78.7%
    # -> Sind die Datensätze zu weit voneinander entfernt? Es scheint so.


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False
    # for layer in model.layers:
    #     layer.trainable = False


def predict_train(model, train_dat, train_lb, val_dat, val_lb, lr, freeze, epochs):

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.fit(train_dat, train_lb, batch_size=128, epochs=epochs, validation_data=(val_dat, val_lb), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    model.pop()


def post_flatten(model, a, b, c, d, lr, freeze):
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    freezing(model, freeze)
    model.pop()


schedule()
# cascade_network()
