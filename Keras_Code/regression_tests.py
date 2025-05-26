import keras
import pandas

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


regression_two()
# regression_one()
