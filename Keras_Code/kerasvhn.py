
import numpy as np
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
    # todo: MNIST_corrupted test
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
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.fit(a, b, batch_size=128, epochs=1, validation_data=(c, d), callbacks=[lr])

    model.summary()
    # todo: make pred as additional input for the 2nd Network!
    pred = model.predict(a)
    print(pred.shape)

    # arr = new_channel(a, pred)
    model_2 = multiple_inputs()
    # model_2 = keras.Sequential([keras.Input(shape=((32, 32, 1), 10))])
    # model_2.add(keras.layers.Flatten())
    # todo: Da wird die erste Dimension abgehackt! - Das soll nicht sein. Geht so nicht. Wäre pred()+bias
    # model_2.add(keras.layers.Dense(units=10, activation='softmax'))
    model_2.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])
    model_2.fit((a, pred), b, batch_size=128, epochs=1, validation_data=(c, d), callbacks=[lr])


def multiple_inputs():
    # define two sets of inputs
    inputA = keras.Input(shape=(32, 32, 1))
    inputB = keras.Input(shape=(10, ))
    # the first branch operates on the first input
    x = keras.layers.Flatten()(inputA)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(10, activation='relu')(x)
    x = keras.Model(inputs=inputA, outputs=x)
    # the second branch operates on the second input
    y = keras.layers.Dense(10, activation="relu")(inputB)
    y = keras.Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = keras.layers.concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = keras.layers.Dense(10, activation="relu")(combined)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.Model(inputs=[x.input, y.input], outputs=z)
    return model


def new_channel(f, pred):  # pred, a):
    # a = a[:, [2, 0, 1]]  # N, 3, 32, 1
    pred = np.expand_dims(pred, axis=0)
    # pred are label predictions, not data
    np.expand_dims(f, axis=0)
    i = 0
    counter = 0
    while i < len(f):
        counter += 1
        i += 1
    print(f.shape)
    print(counter)
    # b = np.concat((a, pred), axis=0)
    # np.concatenate: Input Vectors must have the same dimensions. -> Write after each another
    # np.append: complete Flatten, then append
    # print(b.shape)
    return pred

    '''arr = []
    i = 0
    while i < len(pred):
        arr.append(pred[i])
        i += 1
    arr = np.array(arr)
    print(arr.shape)
    return arr'''

    '''x = np.array([[1, 2]])
    y = np.expand_dims(x, axis=0)
    print(x)
    y[0][0] = 3
    print(y)'''
    '''a_with_pred = np.expand_dims(a, axis=2)
    a_with_pred = a_with_pred[:, [2, 0, 1]]
    i = 0
    while i < len(a_with_pred):
        a_with_pred[i][1] = pred[i]
        i += 1
    return a_with_pred'''


# schedule()
cascade_network()
