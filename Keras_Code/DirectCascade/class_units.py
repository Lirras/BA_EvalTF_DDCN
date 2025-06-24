import keras
import Keras_Code.libraries.metrics as metr


keras.utils.set_random_seed(812)


class Regression():

    def __init__(self):
        super().__init__()

    def initialize(self, in_shape):
        self.network = keras.Sequential([
            keras.Input(shape=in_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])

        self.network.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-03),
                             loss=keras.losses.MeanSquaredError, metrics=['mae'])

    def train(self, train, target, val_tr, val_lb, epochs):
        # return metr.kfoldcross(self.network, train, target, epochs)
        # return metr.mae_met(self.network, train, target, val_tr, val_lb, epochs)
        # return metr.loss_met(self.network, train, target, val_tr, val_lb, epochs)
        hist = self.network.fit(train, target, batch_size=16, epochs=epochs, validation_data=(val_tr, val_lb))
        return hist

    def pred(self, train):
        output = self.network.predict(train)
        return output


class Classification():
    def __init__(self):
        super().__init__()

    def initialize(self, in_shape):
        self.network = keras.Sequential([
            keras.Input(shape=in_shape),
            # keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                             loss=keras.losses.CategoricalCrossentropy, metrics=['accuracy'])

    def train(self, train, target, val_tr, val_lb, epochs):
        # return metr.acc_met(self.network, train, target, val_tr, val_lb, epochs)
        # return metr.loss_met(self.network, train, target, val_tr, val_lb, epochs)
        hist = self.network.fit(train, target, batch_size=128, epochs=epochs, validation_data=(val_tr, val_lb))
        return hist

    def pred(self, train):
        output = self.network.predict(train)
        return output


class LittleConv():
    def __init__(self):
        super().__init__()

    def initialize(self, in_shape):
        self.network = keras.Sequential([
            keras.Input(shape=in_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                             loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    def train(self, train, label, val_tr, val_lb, epochs):
        # return metr.acc_met(self.network, train, label, val_tr, val_lb, epochs)
        # return metr.loss_met(self.network, train, label, val_tr, val_lb, epochs)
        return self.network.fit(train, label, batch_size=128, epochs=epochs, validation_data=(val_tr, val_lb))

    def pred(self, train):
        return self.network.predict(train)


class OneDLilConv():
    def __init__(self):
        super().__init__()

    def initialize(self, in_shape):
        self.network = keras.Sequential([
            keras.Input(shape=in_shape),
            keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                             loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    def train(self, train, label, val_tr, val_lb, epochs):
        # return metr.acc_met(self.network, train, label, val_tr, val_lb, epochs)
        # return metr.loss_met(self.network, train, label, val_tr, val_lb, epochs)
        return self.network.fit(train, label, batch_size=128, epochs=epochs, validation_data=(val_tr, val_lb))

    def pred(self, train):
        return self.network.predict(train)
