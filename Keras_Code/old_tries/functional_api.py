import time
import pandas
import keras
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_regressoion_lib as krl
import Keras_Code.libraries.keras_cascade_lib as kcl
import Keras_Code.libraries.plotting as pltt


def regression_test():
    z1 = time.perf_counter()
    epochs = 20
    name = 'func_regression'
    krl.clear()
    # a, b, c, d = dat_loader.boston_loader()
    e, f, g, h = dat_loader.california_loader()
    lr, optim = krl.lr_optim_reg()

    inputs = keras.Input(shape=(3,))
    inputB = keras.Input(shape=(1,))

    x = keras.layers.Dense(units=1024, activation='relu')(inputs)
    x = keras.layers.Dense(units=512, activation='relu')(x)
    output = keras.layers.Dense(units=1, activation='linear')(x)

    model = keras.Model(inputs, output, name='boston1')

    m = model.predict(e)
    n = model.predict(g)

    # output_2 = keras.layers.Identity()(inputB)
    modelB = keras.Model(inputB, inputB)
    combine = keras.layers.concatenate([model.output, modelB.output])

    x = keras.layers.Dense(units=10, activation='relu')(combine)
    output = keras.layers.Dense(units=1, activation='linear')(x)

    model_c = keras.Model(inputs=[model.input, modelB.input], outputs=output)

    # keras.utils.plot_model(model, 'boston1.png', show_shapes=True)
    # needed pydot and graphviz; --> graphviz doesn't work

    model_c.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['mae'])
    hist_1 = model_c.fit((e, m), f, batch_size=16, epochs=epochs, validation_data=((g, n), h), callbacks=[lr])

    krl.freezing_model(model)
    r = model.predict(e)
    s = model.predict(g)

    hist_2 = model_c.fit((e, r), f, batch_size=16, epochs=epochs, validation_data=((g, s), h), callbacks=[lr])
    z2 = time.perf_counter()
    print(f'{z2-z1:0.2f} sec')
    df_0 = pandas.DataFrame.from_dict(hist_1.history)
    df_1 = pandas.DataFrame.from_dict(hist_2.history)
    pltt.multiple_plots(pltt.add_epoch_counter_to_df(pandas.concat([df_0, df_1])), epochs, len(e), round(z2-z1), name)


def cascade_test():
    z1 = time.perf_counter()
    kcl.clear()
    epochs = 20
    name = 'func_classification'
    # a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()
    lr, optim = kcl.better_lr_optim()

    inputA = keras.Input(shape=(32, 32, 1))
    inputB = keras.Input(shape=(10,))

    x = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inputA)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    outputA = keras.layers.Dense(units=10, activation='softmax')(x)
    model_one = keras.Model(inputA, outputA)

    m = model_one.predict(e)
    n = model_one.predict(g)

    model_two = keras.Model(inputB, inputB)

    x = keras.layers.concatenate([model_one.output, model_two.output])
    x = keras.layers.Dense(units=8192, activation='relu')(x)
    x = keras.layers.Reshape((16, 16, 32))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='relu')(x)

    output = keras.layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model([model_one.input, model_two.input], output)

    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])
    hist_1 = model.fit((e, m), f, batch_size=128, epochs=epochs, validation_data=((g, n), h), callbacks=[lr])

    krl.freezing_model(model_one)
    r = model_one.predict(e)
    s = model_one.predict(g)

    hist_2 = model.fit((e, r), f, batch_size=128, epochs=epochs, validation_data=((g, s), h), callbacks=[lr])
    # 75%/ Better_lr: 18.9%/ little_bet_lr: 79.8%/ ohne TF: 89.7%
    z2 = time.perf_counter()
    print(f'{z2-z1:0.2f} sec')
    df_0 = pandas.DataFrame.from_dict(hist_1.history)
    df_1 = pandas.DataFrame.from_dict(hist_2.history)
    pltt.class_mult_plots(pltt.add_epoch_counter_to_df(pandas.concat([df_0, df_1])), epochs, len(e), round(z2-z1), name)


regression_test()
cascade_test()
