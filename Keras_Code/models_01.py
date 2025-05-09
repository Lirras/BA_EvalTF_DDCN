
import keras
import keras_data_loader
from kerasvhn import clear
from kerasvhn import lr_optim
from kerasvhn import predict_train
from kerasvhn import freezing


def cascade_network():

    clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.mnist_model()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    predict_train(model, e, f, g, h, lr, 0)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, e, f, g, h, lr, 1)

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    predict_train(model, e, f, g, h, lr, 2)

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, e, f, g, h, lr, 3)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, 'softmax'))
    model.fit(e, f, batch_size=128, epochs=4, validation_data=(g, h), callbacks=[lr])

    model.summary()


def schedule():
    batch_size = 128

    keras.backend.clear_session()
    lr_schedule, optimizer = lr_optim()

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
    model.fit(m_tr_dat, m_tr_lb, batch_size=batch_size, epochs=4, validation_data=(m_val_dat, m_val_lb))
    model.pop()
    freezing(model, 0)
    freezing(model, 1)
    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, m_tr_dat, m_tr_lb, m_val_dat, m_val_lb, lr_schedule, 3)
    freezing(model, 2)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, m_tr_dat, m_tr_lb, m_val_dat, m_val_lb, lr_schedule, 4)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 5)

    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 6)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(train_data, train_label, batch_size=batch_size, epochs=4, validation_data=(test_data, test_label))
    model.summary()
