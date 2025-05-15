
# import numpy as np
import tensorflow
import keras
import pandas
import seaborn as sns
import keras_data_loader
import keras_cascade_lib as kcl
# import copied_models
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix


def schedule():
    batch_size = 128
    sns.set_theme()
    keras.backend.clear_session()
    lr_schedule, optimizer = kcl.lr_optim()

    m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = keras_data_loader.mnist_loader()
    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    # df = sns.load_dataset("penguins")
    # df = pandas.DataFrame(m_tr_dat[300].squeeze())
    # sns.pairplot(df)

    # tips = sns.load_dataset('tips')
    '''sns.relplot(
        data=m_tr_dat[300].squeeze(),
    )'''
    # sns.displot(data=m_tr_dat[300].squeeze(), kind='kde')
    # plt.show()

    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(m_tr_dat, m_tr_lb, batch_size=128, epochs=1, validation_data=(m_val_dat, m_val_lb), callbacks=[lr_schedule])
    df_1 = pandas.DataFrame.from_dict(history.history)
    # accuracy loss val_accuracy val_loss learning_rate
    # print(df)
    # sns.pairplot(df, x_vars='val_loss', y_vars='val_accuracy')
    # plt.show()
    kcl.freezing(model, 0)
    kcl.freezing(model, 1)
    model.pop()
    model.add(keras.layers.Reshape((32, 32, 1)))
    # todo: maybe something with Recurrent Layers?
    # model.add(keras.layers.RNN())
    kcl.predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 2, 1)
    # kcl.freezing(model, 3)

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(train_data, train_label, batch_size=128, epochs=10, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    df_2 = pandas.DataFrame.from_dict(history.history)
    # accuracy loss val_accuracy val_loss learning_rate
    # print(df)
    df = pandas.concat([df_1, df_2])
    sns.pairplot(df, vars=('accuracy', 'val_accuracy'), x_vars='val_accuracy', y_vars='accuracy', kind='kde')
    plt.show()
    model.summary()


def cascade_network():

    kcl.clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=1, validation_data=(g, h), callbacks=[lr])

    model.summary()


schedule()
# cascade_network()
