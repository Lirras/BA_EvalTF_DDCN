import keras
import pandas
import seaborn as sns
from matplotlib import pyplot as plt


def lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03, amsgrad=True)
    return lr_schedule, optimizer


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False
    # for layer in model.layers:
    #     layer.trainable = False


def predict_train(model, train_dat, train_lb, val_dat, val_lb, lr, freeze, epochs):

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    history = model.fit(train_dat, train_lb, batch_size=128, epochs=epochs, validation_data=(val_dat, val_lb), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    model.pop()
    return pandas.DataFrame.from_dict(history.history)


def post_flatten(model, a, b, c, d, lr, freeze):
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    return pandas.DataFrame.from_dict(history.history)


def add_epoch_counter_to_df(df):
    sort_list = []
    i = 0
    while i < len(df):
        sort_list.append(i)
        i += 1
    df['epochs'] = sort_list
    return df


def plotting_all(df):
    # If model begins with Conv:
    sns.relplot(df, x='epochs', y='Accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_Accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    plt.show()


def plotting_all_sm(df):
    # If model begins with Dense:
    sns.relplot(df, x='epochs', y='accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    plt.show()
