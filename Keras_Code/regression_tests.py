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
    model.add(keras.layers.Dense(units=64, activation='relu'))
    df_0 = krl.predict(model, e, f, g, h, lr, 20, 16)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation=keras.activations.linear))
    hist = model.fit(e, f, batch_size=16, epochs=20, validation_data=(g, h), callbacks=[lr])
    df_1 = pandas.DataFrame.from_dict(hist.history)
    plot.regression_all(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1])))
    model.summary()
    # TF: 60.4k$ off
    # Clean: 55.2k$ off


regression_one()
