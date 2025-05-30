import keras
import pandas
import numpy
import time

import keras_regressoion_lib as krl
import plotting as plot
import keras_data_loader


def regression_one():
    krl.clear()
    a, b, c, d = keras_data_loader.boston_loader()
    e, f, g, h, = keras_data_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_0 = krl.predict(model, a, b, c, d, lr, 10, 16)
    # mae: 5.4-4.3/5.0-3.6
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_1 = krl.predict(model, a, b, c, d, lr, 5, 16)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_2 = krl.predict(model, e, f, g, h, lr, 10, 16)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation=keras.activations.linear))
    # It goes up to 58, but the graph says up to approximately 78 --> WHY?
    hist = model.fit(e, f, batch_size=16, epochs=2, validation_data=(g, h), callbacks=[lr])
    # mae: 58.0-63.1/57.5-55.6
    df_3 = pandas.DataFrame.from_dict(hist.history)
    # mae_better_lr: 3.5-3.6/56.3-56.2 /NAP: 4.2-4.1/56.5-57.1

    df = plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3]))
    # plot.regression_all(df)
    plot.multiple_plots(df)
    model.summary()
    # TF: 60.4k$ off
    # Clean: 55.2k$ off


def regression_two():
    krl.clear()
    a, b, c, d = keras_data_loader.boston_loader()
    e, f, g, h, = keras_data_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=128, activation='relu'))
    df_0 = krl.predict(model, a, b, c, d, lr, 25, 16)
    model.add(keras.layers.BatchNormalization())
    df_1 = krl.predict(model, a, b, c, d, lr, 25, 16)
    model.add(keras.layers.Dense(units=256, activation='relu'))
    df_2 = krl.predict(model, a, b, c, d, lr, 25, 16)
    model.add(keras.layers.Dropout(0.5))
    df_3 = krl.predict(model, a, b, c, d, lr, 25, 16)

    model.add(keras.layers.Dense(units=512, activation='relu'))
    df_4 = krl.predict(model, e, f, g, h, lr, 25, 16)
    model.add(keras.layers.BatchNormalization())  # What happened here?
    df_5 = krl.predict(model, e, f, g, h, lr, 25, 16)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_6 = krl.predict(model, e, f, g, h, lr, 25, 16)
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    hist = model.fit(e, f, batch_size=16, epochs=25, validation_data=(g, h), callbacks=[lr])

    df_7 = pandas.DataFrame.from_dict(hist.history)

    df = plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7]))
    plot.multiple_plots(df)
    model.summary()


def direct_cascade_reg():
    z1 = time.perf_counter()
    krl.clear()
    a, b, c, d = keras_data_loader.boston_loader()
    # e, f, g, h, = keras_data_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    hist = model.fit(a, b, batch_size=16, epochs=1, validation_data=(c, d), callbacks=[lr])
    pred = model.predict(a)
    pred_val = model.predict(c)
    new_in = krl.build_2nd_in_same(a, pred)
    new_val = krl.build_2nd_in_same(c, pred_val)
    # todo: new_in is the new input for the next Network with the same data.

    # krl.clear()
    model_2 = keras.Sequential([keras.Input(shape=(4,))])
    model_2.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model_2.add(keras.layers.Dense(units=1024, activation='relu'))
    model_2.add(keras.layers.Dense(units=1, activation='linear'))
    hist_2 = model_2.fit(new_in, b, batch_size=16, epochs=1, validation_data=(new_val, d), callbacks=[lr])

    z2 = time.perf_counter()
    df_one = pandas.DataFrame.from_dict(hist.history)
    df_two = pandas.DataFrame.from_dict(hist_2.history)
    plot.multiple_plots(plot.add_epoch_counter_to_df(pandas.concat([df_one, df_two])))
    print(f'time: {z2-z1:0.2f} sec')


direct_cascade_reg()
# regression_two()
# regression_one()
