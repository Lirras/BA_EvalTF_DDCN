
import pandas
import keras
import keras_data_loader
import keras_cascade_lib as kcl
import plotting as plot


def cascade_network():

    kcl.clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

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
    df_1 = kcl.predict_train(model, e, f, g, h, lr, 0, 1)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    df_2 = kcl.predict_train(model, e, f, g, h, lr, 1, 1)

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    df_3 = kcl.predict_train(model, e, f, g, h, lr, 2, 1)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    df_4 = kcl.predict_train(model, e, f, g, h, lr, 3, 2)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, 'softmax'))
    history = model.fit(e, f, batch_size=128, epochs=20, validation_data=(g, h), callbacks=[lr])
    df_5 = pandas.DataFrame.from_dict(history.history)

    df = pandas.concat([df_1, df_2, df_3, df_4, df_5])
    df = plot.add_epoch_counter_to_df(df)
    plot.class_all(df)

    model.summary()


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    lr_schedule, optimizer = kcl.lr_optim()

    m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = keras_data_loader.mnist_loader()
    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    # model = test_model()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(m_tr_dat, m_tr_lb, batch_size=batch_size, epochs=4, validation_data=(m_val_dat, m_val_lb))
    df_0 = pandas.DataFrame.from_dict(history.history)
    model.pop()
    kcl.freezing(model, 0)
    kcl.freezing(model, 1)
    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_1 = kcl.predict_train(model, m_tr_dat, m_tr_lb, m_val_dat, m_val_lb, lr_schedule, 3, 1)
    kcl.freezing(model, 2)
    model.add(keras.layers.BatchNormalization())
    df_2 = kcl.predict_train(model, m_tr_dat, m_tr_lb, m_val_dat, m_val_lb, lr_schedule, 4, 1)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    df_3 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 5, 1)

    model.add(keras.layers.MaxPooling2D((2, 2)))
    df_4 = kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 6, 1)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=4, validation_data=(test_data, test_label))
    df_5 = pandas.DataFrame.from_dict(history.history)
    plot.class_all_sm(plot.add_epoch_counter_to_df(pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5])))
    model.summary()


cascade_network()
