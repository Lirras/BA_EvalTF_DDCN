import keras
import keras_cascade_lib as kcl
import keras_data_loader


def regression_one():
    kcl.clear()
    a, b, c, d = keras_data_loader.boston_loader()
    e, f, g, h, = keras_data_loader.california_loader()
    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(1, 3))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['Accuracy'])
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    model.summary()


regression_one()
