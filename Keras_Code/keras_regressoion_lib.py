import keras
import pandas
import numpy


def lr_optim_reg():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 100 ** (10/(epoch + 10)))
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-03)
    return lr_schedule, optimizer


def predict(model, a, b, c, d, lr, epoch, batch):
    model.add(keras.layers.Dense(units=1, activation=keras.activations.linear))
    history = model.fit(a, b, batch_size=batch, epochs=epoch, validation_data=(c, d), callbacks=[lr])
    model.pop()
    freezing(model, len(model.layers)-1)
    return pandas.DataFrame.from_dict(history.history)


def clear():
    keras.backend.clear_session()


def freezing(model, layer):
    model.layers[layer].trainable = False


def build_2nd_in_same(e, pred):
    i = 0
    ls = []
    while i < len(e):
        ls.append(numpy.concatenate((e[i], pred[i])))
        i += 1
    new_in = numpy.array(ls)
    return new_in
