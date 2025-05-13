
import keras
import keras_data_loader
import keras_cascade_lib as kcl


def Conv8Epochs():
    kcl.clear()

    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()

    lr, optim = kcl.lr_optim()
    model = keras.Sequential([keras.Input(shape=(32, 32, 1))])
    # model = copied_models.mnist_model()
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(a, b, batch_size=128, epochs=2, validation_data=(c, d), callbacks=[lr])
    model.pop()
    model.pop()
    # predict_train(model, e, f, g, h, lr, 0)
    model.add(keras.layers.BatchNormalization())
    kcl.predict_train(model, e, f, g, h, lr, 1, 1)
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    kcl.predict_train(model, e, f, g, h, lr, 2, 8)
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    kcl.predict_train(model, e, f, g, h, lr, 3, 8)
    model.add(keras.layers.BatchNormalization())
    kcl.predict_train(model, e, f, g, h, lr, 4, 1)
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    kcl.predict_train(model, e, f, g, h, lr, 3, 8)
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    kcl.predict_train(model, e, f, g, h, lr, 5, 8)
    model.add(keras.layers.MaxPooling2D(4, 4))
    kcl.predict_train(model, e, f, g, h, lr, 6, 1)
    model.add(keras.layers.Dropout(0.3))
    kcl.predict_train(model, e, f, g, h, lr, 7, 1)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    kcl.post_flatten(model, e, f, g, h, lr, 8)
    kcl.freezing(model, 9)

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.fit(e, f, batch_size=128, epochs=1, validation_data=(g, h), callbacks=[lr])
    # 93.7 TF 11.7 -> 64.4
    # Conv8epochs: 92.7 TF 11.0 -> 68.5

    model.summary()
    # TF: 75.8%
    # ohne: 78.7%
    # -> It seems, that the datasets are to far away from each other.
