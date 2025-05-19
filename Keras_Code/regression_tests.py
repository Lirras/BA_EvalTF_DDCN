import keras
import keras_cascade_lib as kcl
import keras_data_loader


def regression_one():
    kcl.clear()
    keras_data_loader.boston_loader()
    keras_data_loader.california_loader()
    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError, metrics=['Accuracy'])


regression_one()
