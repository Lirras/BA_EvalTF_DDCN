import keras
import numpy as np
import pandas


def lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(  # learning rate grows -> This is senseless.
        lambda epoch: 1e-4 * 10 ** (epoch / 10))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03, amsgrad=True)
    return lr_schedule, optimizer


def better_lr_optim():
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10 ** (10/(epoch + 10))
    )
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


def post_flatten(model, a, b, c, d, lr, freeze, epochs):
    model.add(keras.layers.Dense(10, activation='softmax'))
    history = model.fit(a, b, batch_size=128, epochs=epochs, validation_data=(c, d), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    return pandas.DataFrame.from_dict(history.history)


def build_vec_dense_only(augment, vec, x, in_shape):
    augment = augment.reshape((in_shape, x))  # Das wird immer größer... Ähem...
    ls = []
    i = 0
    while i < len(augment):
        ls.append(np.concat((augment[i], vec[i])))
        i += 1

    neu = np.array(ls)
    return neu


def old_vec_builder(augment, vec):
    # Scheint zu funktionieren, dauert aber ewig, ist also sinnlos. Es muss schneller gehen.
    # Here comes a memory Error
    ls = []
    a = 0
    while a < len(augment):
        f = []
        b = 0
        while b < len(augment[a]):  # 32
            c = 0
            g = []
            while c < len(augment[a][b]):  # 32
                d = 0
                h = []
                while d < len(augment[a][b][c]):  # 1
                    e = 0
                    i = []
                    while e < len(vec[a]):  # 10
                        h.append([augment[a][b][c][d], vec[a][e]])
                        e += 1
                    h.append(i)
                    d += 1
                g.append(h)
                c += 1
            f.append(g)
            b += 1
        ls.append(f)
        a += 1

    arr_2 = np.array(ls)
    print(arr_2.shape)

    return arr_2


def build_vec_conv(augment, vec):
    # This is functioning, but computation takes a minute!
    ls2 = []
    i = 0
    while i < len(vec):
        j = 0
        ls = []
        while j < len(vec[i]):
            ls.append(np.full(shape=(1, 32, 32), fill_value=vec[i][j]))
            j += 1
        ls2.append(np.concat(ls))
        i += 1
    vec_arr = np.array(ls2)
    transpose_vec = np.transpose(vec_arr, (0, 2, 3, 1))
    concat_vec = np.array(transpose_vec)

    q = 0
    end = []
    while q < len(augment):
        end.append(np.concat((augment[q], concat_vec[q]), axis=2))
        q += 1
    end = np.array(end)
    # Unable to allocate 168 KiB for array with Shape 32 32 21 and dtype float64, but dataset reduction could fix this
    return end


def numpy_arrs():
    arr1 = np.array([[1, 2, 3], [8, 9, 0]])
    arr2 = np.array([[4, 5, 6, 7]])
    arr = np.concatenate((arr1, arr2))  # This one, cause no changes.
    print(arr)  # 6
    arr = np.hstack((arr1, arr2))  # maybe
    print(arr)  # 6
    arr = np.stack((arr1, arr2), axis=1)  # problems
    print(arr)  # 3 2
    arr = np.vstack((arr1, arr2))  # problems
    print(arr)  # 2 3
    arr = np.dstack((arr1, arr2))  # problems
    print(arr)  # 3 2
