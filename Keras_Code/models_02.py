import keras
import keras_data_loader
from kerasvhn import predict_train
from kerasvhn import freezing
from kerasvhn import lr_optim
from kerasvhn import clear
from kerasvhn import post_flatten


def dropout_model():
    clear()
    m_tr_dat, m_tr_lb, m_val_dat, m_val_lb = keras_data_loader.mnist_loader()
    train_data, train_label, test_data, test_label = keras_data_loader.svhn_loader()

    lr_schedule, optimizer = lr_optim()

    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.svhn_model()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.add(keras.layers.Reshape((32, 32, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 3, 1)
    freezing(model, 2)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 4, 1)
    model.add(keras.layers.Dropout(0.5))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 5, 1)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 6, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 7, 1)
    model.add(keras.layers.Dropout(0.5))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 8, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 9, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 10, 1)
    model.add(keras.layers.Dropout(0.5))
    predict_train(model, train_data, train_label, test_data, test_label, lr_schedule, 11, 1)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.fit(train_data, train_label, batch_size=128, epochs=5, validation_data=(test_data, test_label), callbacks=[lr_schedule])
    model.summary()


def mnist_svhn_net():
    clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    # MNIST solver
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    predict_train(model, a, b, c, d, lr, 0, 1)
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, a, b, c, d, lr, 1, 1)
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    predict_train(model, a, b, c, d, lr, 2, 1)
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    predict_train(model, a, b, c, d, lr, 3, 1)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.fit(a, b, batch_size=128, epochs=1, validation_data=(c, d), callbacks=[lr])
    freezing(model, 4)
    freezing(model, 5)
    model.pop()
    model.add(keras.layers.Reshape((48, 48, 1)))

    # SVHN Solver
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 6, 10)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 7, 1)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 8, 5)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 9, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 10, 4)

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 11, 5)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 12, 1)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 13, 5)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 14, 2)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 15, 2)

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 16, 5)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 17, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 18, 6)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 19, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 20, 2)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=5, validation_data=(g, h), callbacks=[lr])
    model.pop()
    freezing(model, 21)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=5, validation_data=(g, h), callbacks=[lr])
    model.pop()
    freezing(model, 22)
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=3, validation_data=(g, h), callbacks=[lr])
    model.pop()
    freezing(model, 23)
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=8, validation_data=(g, h), callbacks=[lr])

    model.summary()


def batch_norm_net():
    clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, a, b, c, d, lr, 0, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 1, 1)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 2, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 3, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 4, 1)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 5, 1)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 6, 1)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 7, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 8, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 9, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 10, 1)
    model.add(keras.layers.BatchNormalization())
    predict_train(model, e, f, g, h, lr, 11, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    predict_train(model, e, f, g, h, lr, 12, 1)
    model.add(keras.layers.MaxPooling2D((2, 2)))
    predict_train(model, e, f, g, h, lr, 13, 1)
    model.add(keras.layers.Dropout(0.3))
    predict_train(model, e, f, g, h, lr, 14, 1)
    model.add(keras.layers.Flatten())
    freezing(model, 15)
    model.add(keras.layers.Dense(128, activation='relu'))
    post_flatten(model, e, f, g, h, lr, 16)
    model.add(keras.layers.Dropout(0.4))
    post_flatten(model, e, f, g, h, lr, 17)

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=1, validation_data=(g, h), callbacks=[lr])
    model.summary()
