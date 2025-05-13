import keras


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

    model.fit(train_dat, train_lb, batch_size=128, epochs=epochs, validation_data=(val_dat, val_lb), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
    model.pop()


def post_flatten(model, a, b, c, d, lr, freeze):
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    freezing(model, freeze)
    model.pop()
