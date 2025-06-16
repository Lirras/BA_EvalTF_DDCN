import keras
import pandas
import time
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_cascade_lib as kcl
import Keras_Code.libraries.plotting as plot


def dropout_model():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Dropout'
    # m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = dat_loader.mnist_loader()
    train_data, train_label, test_data, test_label = dat_loader.svhn_loader()

    lr_schedule, optimizer = kcl.lr_optim()

    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.svhn_model()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(train_data, train_label, batch_size=128, epochs=2, validation_data=(test_data, test_label))
    df_x = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 0)
    kcl.freezing(model, 1)
    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_0 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 3, epochs)
    kcl.freezing(model, 2)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_1 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 4, epochs)
    model.add(keras.layers.Dropout(0.5))
    df_2 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 5, epochs)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_3 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 6, epochs)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_4 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 7, epochs)
    model.add(keras.layers.Dropout(0.5))
    df_5 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 8, epochs)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_6 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 9, epochs)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_7 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 10, epochs)
    model.add(keras.layers.Dropout(0.5))
    df_8 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 11, epochs)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    hist = model.fit(train_data, train_label, batch_size=128, epochs=epochs, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    z2 = time.perf_counter()
    df_9 = pandas.DataFrame.from_dict(hist.history)
    print(df_9)
    plot.class_all_sm(plot.add_epoch_counter_to_df(pandas.concat([df_x, df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9])), epochs, len(train_data), round(z2-z1), name)
    model.summary()


def mnist_svhn_net():
    kcl.clear()

    z1 = time.perf_counter()
    epochs_mnist = 10
    epochs_svhn = 10
    name = 'MNIST2SVHN'

    # a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    # MNIST solver
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))  # Netz 1
    df_0 = kcl.predict_train(model, e, f, g, h, lr, 0, epochs_mnist)  # Training
    # Inferenz: Trainingsdatensatz mit prediction.
    # Targets sind der Output + Inputall. (als weiteren Bias, channel)
    # todo: model.predict(input) für Output vom Netz, output + input = input_next, dann iterieren
    # todo: Ganze Netze trainieren und dann TF. Enhance Input.
    # todo: Zeitmesser.
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Netz 2
    df_1 = kcl.predict_train(model, e, f, g, h, lr, 1, epochs_mnist)  # Training
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    df_2 = kcl.predict_train(model, e, f, g, h, lr, 2, epochs_mnist)
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    df_3 = kcl.predict_train(model, e, f, g, h, lr, 3, epochs_mnist)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    history = model.fit(e, f, batch_size=128, epochs=epochs_mnist, validation_data=(g, h), callbacks=[lr])
    df_4 = pandas.DataFrame.from_dict(history.history)
    kcl.freezing(model, 4)
    kcl.freezing(model, 5)
    model.pop()
    model.add(keras.layers.Reshape((48, 48, 1)))

    # SVHN Solver
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_5 = kcl.predict_train(model, e, f, g, h, lr, 6, epochs_svhn)
    model.add(keras.layers.BatchNormalization())
    df_6 = kcl.predict_train(model, e, f, g, h, lr, 7, epochs_svhn)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_7 = kcl.predict_train(model, e, f, g, h, lr, 8, epochs_svhn)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_8 = kcl.predict_train(model, e, f, g, h, lr, 9, epochs_svhn)
    model.add(keras.layers.Dropout(0.3))
    df_9 = kcl.predict_train(model, e, f, g, h, lr, 10, epochs_svhn)

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_10 = kcl.predict_train(model, e, f, g, h, lr, 11, epochs_svhn)
    model.add(keras.layers.BatchNormalization())
    df_11 = kcl.predict_train(model, e, f, g, h, lr, 12, epochs_svhn)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_12 = kcl.predict_train(model, e, f, g, h, lr, 13, epochs_svhn)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_13 = kcl.predict_train(model, e, f, g, h, lr, 14, epochs_svhn)
    model.add(keras.layers.Dropout(0.3))
    df_14 = kcl.predict_train(model, e, f, g, h, lr, 15, epochs_svhn)

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_15 = kcl.predict_train(model, e, f, g, h, lr, 16, epochs_svhn)
    model.add(keras.layers.BatchNormalization())
    df_16 = kcl.predict_train(model, e, f, g, h, lr, 17, epochs_svhn)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_17 = kcl.predict_train(model, e, f, g, h, lr, 18, epochs_svhn)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_18 = kcl.predict_train(model, e, f, g, h, lr, 19, epochs_svhn)
    model.add(keras.layers.Dropout(0.3))
    df_19 = kcl.predict_train(model, e, f, g, h, lr, 20, epochs_svhn)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs_svhn, validation_data=(g, h), callbacks=[lr])
    df_20 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 21)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs_svhn, validation_data=(g, h), callbacks=[lr])
    df_21 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 22)
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs_svhn, validation_data=(g, h), callbacks=[lr])
    df_22 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 23)
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs_svhn, validation_data=(g, h), callbacks=[lr])
    z2 = time.perf_counter()
    df_23 = pandas.DataFrame.from_dict(history.history)
    print(df_23)
    plot.class_all(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8,
                                                                df_9, df_10, df_11, df_12, df_13, df_14, df_15, df_16,
                                                                df_17, df_18, df_19, df_20, df_21, df_22, df_23])), epochs_svhn, len(e), round(z2-z1), name)
    model.summary()


def batch_norm_net():
    kcl.clear()

    z1 = time.perf_counter()
    epochs = 10
    name = 'BatchNorm'

    # a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_0 = kcl.predict_train(model, e, f, g, h, lr, 0, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_1 = kcl.predict_train(model, e, f, g, h, lr, 1, epochs)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_2 = kcl.predict_train(model, e, f, g, h, lr, 2, epochs)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_3 = kcl.predict_train(model, e, f, g, h, lr, 3, epochs)
    model.add(keras.layers.Dropout(0.3))
    df_4 = kcl.predict_train(model, e, f, g, h, lr, 4, epochs)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_5 = kcl.predict_train(model, e, f, g, h, lr, 5, epochs)
    model.add(keras.layers.BatchNormalization())
    df_6 = kcl.predict_train(model, e, f, g, h, lr, 6, epochs)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    df_7 = kcl.predict_train(model, e, f, g, h, lr, 7, epochs)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_8 = kcl.predict_train(model, e, f, g, h, lr, 8, epochs)
    model.add(keras.layers.Dropout(0.3))
    df_9 = kcl.predict_train(model, e, f, g, h, lr, 9, epochs)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_10 = kcl.predict_train(model, e, f, g, h, lr, 10, epochs)
    model.add(keras.layers.BatchNormalization())
    df_11 = kcl.predict_train(model, e, f, g, h, lr, 11, epochs)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    df_12 = kcl.predict_train(model, e, f, g, h, lr, 12, epochs)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_13 = kcl.predict_train(model, e, f, g, h, lr, 13, epochs)
    model.add(keras.layers.Dropout(0.3))
    df_14 = kcl.predict_train(model, e, f, g, h, lr, 14, epochs)
    model.add(keras.layers.Flatten())
    kcl.freezing(model, 15)
    model.add(keras.layers.Dense(128, activation='relu'))
    df_15 = kcl.post_flatten(model, e, f, g, h, lr, 16, epochs)
    model.add(keras.layers.Dropout(0.4))
    df_16 = kcl.post_flatten(model, e, f, g, h, lr, 1, epochs)

    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(e, f, batch_size=128, epochs=epochs, validation_data=(g, h), callbacks=[lr])
    z2 = time.perf_counter()
    df_17 = pandas.DataFrame.from_dict(history.history)
    plot.class_all(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7,
                                                                df_8, df_9, df_10, df_11, df_12, df_13, df_14, df_15,
                                                                df_16, df_17])), epochs, len(e), round(z2-z1), name)
    model.summary()


dropout_model()
mnist_svhn_net()
batch_norm_net()
