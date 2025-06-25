import time
import pandas
import keras
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_cascade_lib as kcl
import Keras_Code.libraries.plotting as plot


keras.utils.set_random_seed(812)


def cascade_network(percentage):

    kcl.clear()
    z1 = time.perf_counter()
    name = 'Conv_MaxPool'
    test_ls = []

    a, b, c, d, sts_tr, sts_lb = dat_loader.mnist_loader()
    e, f, g, h, tts_tr, tts_lb = dat_loader.svhn_loader(percentage)
    epochs = 10

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    '''model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        predict_train(model, a, b, c, d, lr, 0)
        model.add(keras.layers.AvgPool2D((2, 2)))
        predict_train(model, a, b, c, d, lr, 1)
        model.add(keras.layers.Flatten())'''

    # avgPool2D: 92.4%/70.4%
    # maxPool2D: 93%/71.9%

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    df_1, pred = kcl.predict_train(model, a, b, c, d, lr, 0, epochs, tts_tr)
    test_ls.append(plot.preds_for_plots(pred, tts_lb))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    df_2, pred = kcl.predict_train(model, a, b, c, d, lr, 1, epochs, tts_tr)
    test_ls.append(plot.preds_for_plots(pred, tts_lb))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    df_3, pred = kcl.predict_train(model, e, f, g, h, lr, 2, epochs, tts_tr)
    test_ls.append(plot.preds_for_plots(pred, tts_lb))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # df_4, pred = kcl.predict_train(model, e, f, g, h, lr, 3, epochs, tts_tr)
    # test_ls.append(plot.preds_for_plots(pred, tts_lb))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, 'softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    test_ls.append(plot.preds_for_plots(model.predict(tts_tr), tts_lb))

    df_5 = pandas.DataFrame.from_dict(history.history)

    z2 = time.perf_counter()

    df = pandas.concat([df_1, df_2, df_3, df_5])  # df_4 were deleted
    df = plot.add_epoch_counter_to_df(df)
    df_x = plot.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_ls}))
    plot.class_networks(df_x, epochs, len(e), round(z2-z1), name)
    plot.class_all(df, epochs, len(e), round(z2-z1), name)

    # model.summary()
    # return df, df_x, len(e)


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    epo_wtf = 1
    epochs = 10
    name = 'Dense2Conv'
    z1 = time.perf_counter()
    lr_schedule, optimizer = kcl.lr_optim()

    # m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = dat_loader.mnist_loader()
    train_data, train_label, test_data, test_label = dat_loader.svhn_loader()

    # model = test_model()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epo_wtf, validation_data=(test_data, test_label))
    df_0 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 0)
    kcl.freezing(model, 1)
    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_1 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 3, epo_wtf)
    kcl.freezing(model, 2)
    model.add(keras.layers.BatchNormalization())
    df_2 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 4, epo_wtf)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_3 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 5, epochs)

    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_4 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 6, epochs)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=4, validation_data=(test_data, test_label))
    z2 = time.perf_counter()
    df_5 = pandas.DataFrame.from_dict(history.history)
    plot.class_all_sm(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5])), epochs, len(train_data), round(z2-z1), name)
    model.summary()


def convmaxpool_complete():
    kcl.clear()
    z1 = time.perf_counter()
    name = 'Conv_MaxPool'
    test_ls = []

    # a, b, c, d, sts_tr, sts_lb = dat_loader.mnist_loader()
    e, f, g, h, tts_tr, tts_lb = dat_loader.svhn_loader()
    epochs = 10

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1)),
                              keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                              keras.layers.MaxPooling2D(pool_size=(2, 2)),
                              keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                              keras.layers.MaxPooling2D(pool_size=(2, 2)),
                              keras.layers.Flatten(),
                              keras.layers.Dense(10, 'softmax')
                              ])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    history = model.fit(e, f, batch_size=128, epochs=40, validation_data=(g, h))  # , callbacks=[lr])
    test_ls.append(plot.preds_for_plots(model.predict(tts_tr), tts_lb))

    df_5 = pandas.DataFrame.from_dict(history.history)

    z2 = time.perf_counter()

    df = pandas.concat([df_5])
    df = plot.add_epoch_counter_to_df(df)
    plot.class_networks(plot.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_ls})), epochs, len(e),
                        round(z2 - z1), name)
    plot.class_all(df, epochs, len(e), round(z2 - z1), name)

    model.summary()


# convmaxpool_complete()
cascade_network(0.3)
# schedule()
