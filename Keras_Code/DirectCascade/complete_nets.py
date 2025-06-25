import time
import pandas
import keras
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_cascade_lib as kcl
import Keras_Code.libraries.plotting as pltt
import Keras_Code.libraries.keras_regressoion_lib as krl


keras.utils.set_random_seed(812)


def TwoDConv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'little_conv'
    e, f, g, h, test_dat_svhn, test_lb_svhn = dat_loader.svhn_loader()

    network = keras.Sequential([
        keras.Input(shape=(32, 32, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    hist = network.fit(e, f, batch_size=128, epochs=40, validation_data=(g, h))
    z2 = time.perf_counter()
    pltt.class_all(pltt.add_epoch_counter_to_df(pandas.DataFrame.from_dict(hist.history)), epochs, len(e),
                   round(z2-z1), name)


def OneDConv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'OneDLilConv'
    e, f, g, h, test_dat_svhn, test_lab_svhn = dat_loader.svhn_loader(0.99)
    e = e.reshape((len(e), 1024, 1))
    g = g.reshape((len(g), 1024, 1))
    x = 1024

    network = keras.Sequential([
        keras.Input(shape=(x, 1)),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])
    hist = network.fit(e, f, batch_size=128, epochs=300, validation_data=(g, h))
    z2 = time.perf_counter()
    pltt.class_all(pltt.add_epoch_counter_to_df(pandas.DataFrame.from_dict(hist.history)), epochs, len(e),
                   round(z2 - z1), name)


def ClassOneDense():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    e, f, g, h, svhn_test, svhn_test_lab = dat_loader.svhn_loader(0.99)
    x = 1024
    e = e.reshape(len(e), x)
    g = g.reshape(len(g), x)

    network = keras.Sequential([
        keras.Input(shape=(x,)),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),

        keras.layers.Dense(units=10, activation='softmax')
    ])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.CategoricalCrossentropy, metrics=['accuracy'])

    hist = network.fit(e, f, batch_size=128, epochs=300, validation_data=(g, h))
    z2 = time.perf_counter()
    pltt.class_all_sm(pltt.add_epoch_counter_to_df(pandas.DataFrame.from_dict(hist.history)), epochs, len(e),
                      round(z2 - z1), name)


def OneLayer():
    krl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Regression_one_Layer'
    e, f, g, h, cali_test_data, cali_test_target = dat_loader.california_loader()

    network = keras.Sequential([
        keras.Input(shape=(3,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    network.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-03),
                    loss=keras.losses.MeanSquaredError, metrics=['mae'])

    hist = network.fit(e, f, batch_size=16, epochs=80, validation_data=(g, h))
    z2 = time.perf_counter()
    pltt.multiple_plots(pltt.add_epoch_counter_to_df(pandas.DataFrame.from_dict(hist.history)), epochs, len(e),
                        round(z2 - z1), name)


# TwoDConv()
print('first')
OneDConv()  # 55min
print('SEcond')
ClassOneDense()  # 35min
print('Third')
# OneLayer()
print('Fourth')
