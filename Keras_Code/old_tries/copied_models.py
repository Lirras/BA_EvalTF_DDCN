# SVHN is Copied from https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc
# MNIST is Copied from https://keras.io/examples/vision/mnist_convnet/

import keras


def svhn_model():
    model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation='relu',
                                input_shape=(32, 32, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), padding='same',
                                activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation='softmax')
        ])
    return model


def mnist_model():
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")])
    return model
