import time
import pandas
import Keras_Code.libraries.plotting as pltt
import keras
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_cascade_lib as kcl
import Keras_Code.libraries.keras_regressoion_lib as krl


def classification():
    z1 = time.perf_counter()
    kcl.clear()
    epochs = 10
    name = 'Name'
    a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()
    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])
    # todo: Write here the Network
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    hist = model.fit(a, b, batch_size=128, epochs=epochs, validation_data=(c, d), callbacks=[lr])
    df_one = pandas.DataFrame.from_dict(hist.history)
    pred = model.predict(a)
    # todo: Make pred as input for 2nd
    # todo: Write here the 2nd Network
    z2 = time.perf_counter()
    print(f'time {z2-z1:0.2f} sec')
    pltt.class_all(pltt.add_epoch_counter_to_df(pandas.concat([df_one])), epochs, len(e), round(z2-z1), name)


def regression():
    z1 = time.perf_counter()
    krl.clear()
    epochs = 10
    name = 'Name'
    a, b, c, d = dat_loader.boston_loader()
    e, f, g, h = dat_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    # todo: Write here the Network

    model.add(keras.layers.Dense(units=1, activation='linear'))
    hist = model.fit(a, b, batch_size=16, epochs=epochs, validation_data=(c, d), callbacks=[lr])
    df_one = pandas.DataFrame.from_dict(hist.history)
    pred = model.predict(a)
    # todo: Make pred as input for 2nd
    new_in = krl.build_2nd_in_same(a, pred)

    # todo: Write here the 2nd Network

    z2 = time.perf_counter()
    print(f'time {z2 - z1:0.2f} sec')
    pltt.multiple_plots(pltt.add_epoch_counter_to_df(pandas.concat([df_one])), epochs, len(e), round(z2-z1), name)
