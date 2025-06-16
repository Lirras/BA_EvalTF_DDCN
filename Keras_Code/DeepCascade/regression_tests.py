import keras
import pandas
import numpy
import time
# import fix_models_test
import tensorflow as tf

import Keras_Code.libraries.keras_regressoion_lib as krl
import Keras_Code.libraries.plotting as plot
import Keras_Code.libraries.keras_data_loader as dat_loader


def regression_one():
    krl.clear()
    # a, b, c, d = dat_loader.boston_loader()
    e, f, g, h, = dat_loader.california_loader()
    epochs = 10
    batch_size = 16
    name = 'Only_Dense'
    z1 = time.perf_counter()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_0 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    # mae: 5.4-4.3/5.0-3.6
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_1 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_2 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation=keras.activations.linear))
    # It goes up to 58, but the graph says up to approximately 78 --> WHY?
    hist = model.fit(e, f, batch_size=batch_size, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    # mae: 58.0-63.1/57.5-55.6
    z2 = time.perf_counter()
    df_3 = pandas.DataFrame.from_dict(hist.history)
    # mae_better_lr: 3.5-3.6/56.3-56.2 /NAP: 4.2-4.1/56.5-57.1

    df = plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3]))
    # plot.regression_all(df)
    plot.multiple_plots(df, epochs, len(e), round(z2-z1), name)
    model.summary()
    # TF: 60.4k$ off
    # Clean: 55.2k$ off


def regression_two():
    krl.clear()
    # a, b, c, d = dat_loader.boston_loader()
    e, f, g, h, = dat_loader.california_loader()
    epochs = 25
    batch_size = 16
    name = 'regression_two'
    z1 = time.perf_counter()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=128, activation='relu'))
    df_0 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.BatchNormalization())
    df_1 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dense(units=256, activation='relu'))
    df_2 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dropout(0.5))
    df_3 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)

    model.add(keras.layers.Dense(units=512, activation='relu'))
    df_4 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.BatchNormalization())  # What happened here?
    df_5 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    df_6 = krl.predict(model, e, f, g, h, lr, epochs, batch_size)
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    hist = model.fit(e, f, batch_size=batch_size, epochs=epochs, validation_data=(g, h), callbacks=[lr])

    z2 = time.perf_counter()
    df_7 = pandas.DataFrame.from_dict(hist.history)

    df = plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7]))
    plot.multiple_plots(df, epochs, len(e), round(z2-z1), name)
    model.summary()


def direct_cascade_reg():
    z1 = time.perf_counter()
    epochs = 40
    batch_size = 16
    name = 'CasInOneReg_FineTune'
    krl.clear()
    # a, b, c, d = dat_loader.boston_loader()
    e, f, g, h, = dat_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(units=512, activation='relu'))
    model.add(keras.layers.Dense(units=256, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='linear'))

    # todo: Why change the data after one fit-call?
    hist = model.fit(e, f, batch_size=batch_size, epochs=40, validation_data=(g, h), callbacks=[lr])
    krl.freezing_model(model)
    # if freezing the Net turned into prediction Mode and give the prediction to the next model,
    # even if it's the same network

    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(units=512, activation='relu'))
    model.add(keras.layers.Dense(units=256, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    hist_2 = model.fit(e, f, batch_size=batch_size, epochs=40, validation_data=(g, h), callbacks=[lr])

    krl.unfreezing_all(model)
    lr, optim = krl.fine_tuning()
    # model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    hist_3 = model.fit(e, f, batch_size=batch_size, epochs=100, validation_data=(g, h), callbacks=[lr])

    '''pred = model.predict(a)
    pred_val = model.predict(c)
    new_in = krl.build_2nd_in_same(a, pred)
    new_val = krl.build_2nd_in_same(c, pred_val)'''
    # todo: new_in is the new input for the next Network with the same data.

    # krl.clear()
    '''model_2 = keras.Sequential([keras.Input(shape=(3,))])
    model_2.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model_2.add(keras.layers.Dense(units=1024, activation='relu'))
    model_2.add(keras.layers.Dense(units=1, activation='linear'))
    hist_2 = model_2.fit(a, b, batch_size=16, epochs=1, validation_data=(c, d), callbacks=[lr])'''
    # todo: all input arrays are not numpy and it cant convert into it! The call for model is the same!
    # todo: Understand the Difference between this two!

    z2 = time.perf_counter()
    df_one = pandas.DataFrame.from_dict(hist.history)
    df_two = pandas.DataFrame.from_dict(hist_2.history)
    df_three = pandas.DataFrame.from_dict(hist_3.history)
    plot.multiple_plots(plot.add_epoch_counter_to_df(pandas.concat([df_one, df_two, df_three])), epochs, len(e), round(z2-z1), name)
    print(f'time: {z2-z1:0.2f} sec')


def dcr():
    print(tf.executing_eagerly())
    z1 = time.perf_counter()
    krl.clear()
    a, b, c, d = dat_loader.boston_loader()
    r, s, t, u = a.copy(), b.copy(), c.copy(), d.copy()
    e, f, g, h, = dat_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = model_01()
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    # todo: Why change the data after one fit-call?
    hist = model.fit(a, b, batch_size=16, epochs=1)
    print(tf.executing_eagerly())
    # Why will it the self call? - But it's an Object instance. Its has no self!
    # That doesn't work too!
    # keras.saving.save_model(model, 'D:/Uni/Sommersemester 2024/Bachelor_Arbeit/Code/cascadetest/BA_EvalTF_DDCN/Keras_Code/test.keras')
    # model.save('D:/Uni/Sommersemester 2024/Bachelor_Arbeit/Code/cascadetest/BA_EvalTF_DDCN/Keras_Code/test.keras')
    # krl.freezing_model(model)

    '''pred = model.predict(a)
    pred_val = model.predict(c)
    new_in = krl.build_2nd_in_same(a, pred)
    new_val = krl.build_2nd_in_same(c, pred_val)'''

    model_2 = model_01()
    model_2.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    print(tf.executing_eagerly())
    hist_2 = model_2.fit(r, s, batch_size=16, epochs=1)

    z2 = time.perf_counter()
    df_one = pandas.DataFrame.from_dict(hist.history)
    df_two = pandas.DataFrame.from_dict(hist_2.history)
    plot.multiple_plots(plot.add_epoch_counter_to_df(pandas.concat([df_one, df_two])))
    print(f'time: {z2 - z1:0.2f} sec')


def late_idea():
    krl.clear()
    a, b, c, d = dat_loader.boston_loader()
    e, f, g, h = dat_loader.california_loader()
    lr, optim = krl.lr_optim_reg()
    model = keras.Sequential([keras.Input(shape=(3,))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    model.add(keras.layers.Dense(units=10, activation='relu'))
    edge = keras.layers.Dense(units=1, activation='linear')
    model.add(edge)
    model.fit(a, b, batch_size=16, epochs=1, validation_data=(c, d), callbacks=[lr])
    krl.freezing_model(model)
    r = model.predict(e)
    s = model.predict(g)
    # model.layers[-1].output[1]
    # model.outputs[0][1] --> (1,), (None, 1) // Ohne 1 am Ende: (None, 2), (None, 1)
    model.add(keras.layers.concatenate([edge, keras.Input(shape=(1,))]))
    # Only Instances of keras.layers can be added / Only input tensors may be passed as positional arguments.
    # --> Need keras.layers as key argument
    model.add(keras.layers.Dense(units=10, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    model.fit(e, f, batch_size=16, epochs=1, validation_data=(g, h), callbacks=[lr])
    # todo: Das ganze mal als Klasse und dessen instanzen austesten -> PyDCA


class model_01(keras.Sequential):
    def __init__(self):
        super().__init__()


# late_idea()
# dcr()
direct_cascade_reg()
regression_two()
regression_one()
