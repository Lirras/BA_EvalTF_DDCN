
import pandas
import keras
import time
import keras_data_loader
import keras_cascade_lib as kcl
import plotting as plot


def Conv8Epochs():
    kcl.clear()

    z1 = time.perf_counter()
    epochs = 10
    name = 'Conv8Epochs'

    # a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    df_0 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    model.pop()
    model.add(keras.layers.BatchNormalization())
    df_1 = kcl.predict_train(model, e, f, g, h, lr, 1, epochs)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_2 = kcl.predict_train(model, e, f, g, h, lr, 2, epochs)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_3 = kcl.predict_train(model, e, f, g, h, lr, 3, epochs)
    model.add(keras.layers.BatchNormalization())
    df_4 = kcl.predict_train(model, e, f, g, h, lr, 4, epochs)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_5 = kcl.predict_train(model, e, f, g, h, lr, 3, epochs)
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    df_6 = kcl.predict_train(model, e, f, g, h, lr, 5, epochs)
    model.add(keras.layers.MaxPooling2D(4, 4))
    df_7 = kcl.predict_train(model, e, f, g, h, lr, 6, epochs)
    model.add(keras.layers.Dropout(0.3))
    df_8 = kcl.predict_train(model, e, f, g, h, lr, 7, epochs)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    df_9 = kcl.post_flatten(model, e, f, g, h, lr, 8, epochs)
    kcl.freezing(model, 9)

    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    df_10 = pandas.DataFrame.from_dict(history.history)
    z2 = time.perf_counter()
    plot.class_all(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7,
                                                                df_8, df_9, df_10])), epochs, len(e), round(z2-z1), name)
    # 93.7 TF 11.7 -> 64.4
    # Conv8epochs: 92.7 TF 11.0 -> 68.5

    model.summary()
    # TF: 75.8%
    # ohne: 78.7%
    # -> It seems, that the datasets are too far away from each other.


def Lin2Layer():
    kcl.clear()

    epochs = 10
    name = 'twoLinLayer'
    z1 = time.perf_counter()

    # a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    df_0 = kcl.post_flatten(model, e, f, g, h, lr, 0, epochs)
    kcl.freezing(model, 1)
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    df_1 = pandas.DataFrame.from_dict(history.history)
    z2 = time.perf_counter()
    plot.class_all(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1])), epochs, len(e), round(z2-z1), name)
    model.summary()


Lin2Layer()
Conv8Epochs()
