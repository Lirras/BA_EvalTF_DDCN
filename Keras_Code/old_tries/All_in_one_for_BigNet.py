import pandas
import keras
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def svhn_loader():

    train, test = load()
    train_img = np.array(train['X'])
    test_img = np.array(test['X'])

    train_labels = train['y']
    test_labels = test['y']

    print(train_img.shape)  # 32 32 3 73257
    print(test_img.shape)  # 32 32 3 26032

    train_img = np.moveaxis(train_img, -1, 0)  # B, H, W, C
    test_img = np.moveaxis(test_img, -1, 0)  # B, H, W, C

    # GrayScaling for compatibility with mnist
    train_img = np.dot(train_img[..., :3], [0.2989, 0.5870, 0.1140])
    test_img = np.dot(test_img[..., :3], [0.2989, 0.5870, 0.1140])

    train_img = np.expand_dims(train_img, -1)
    test_img = np.expand_dims(test_img, -1)

    print(train_img.shape)
    print(test_img.shape)

    print('Min: {}, Max: {}'.format(train_img.min(), train_img.max()))

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    X_train, xval, y_train, yval = train_test_split(train_img, train_labels, test_size=0.99, random_state=22)
    x, X_val, y, y_val = train_test_split(train_img, train_labels, test_size=0.01, random_state=22)
    # print(X_train.shape)
    # print(y_train.shape)

    # less data
    # train_img = X_train.astype('float64')
    # train_labels = y_train.astype('int64')
    # test_img = X_val.astype('float64')
    # test_labels = y_val.astype('int64')

    # full data
    train_img = train_img.astype('float64')
    test_img = test_img.astype('float64')

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    return train_img, train_labels, test_img, test_labels


def mnist_loader():
    (train_dat, train_lb), (val_dat, val_lb) = keras.datasets.mnist.load_data()

    # Upscaling for compatibility with svhn
    train_dat = np.pad(train_dat, ((0, 0), (0, 4), (0, 4)), 'constant')
    val_dat = np.pad(val_dat, ((0, 0), (0, 4), (0, 4)), 'constant')

    # Scale images to the [0, 1] range
    train_dat = train_dat.astype("float32") / 255
    val_dat = val_dat.astype("float32") / 255
    # Make sure images have shape (32, 32, 1)
    train_dat = np.expand_dims(train_dat, -1)
    val_dat = np.expand_dims(val_dat, -1)

    print("train_dat shape:", train_dat.shape)
    print(train_dat.shape[0], "train samples")
    print(train_lb.shape[0], "label samples")

    lb = LabelBinarizer()
    train_lb = lb.fit_transform(train_lb)
    val_lb = lb.fit_transform(val_lb)

    val_lb = val_lb.astype('int64')
    train_lb = train_lb.astype('int64')

    return train_dat, train_lb, val_dat, val_lb


def load():
    train = loadmat("D:/SVHN/train_32x32.mat")
    test = loadmat("D:/SVHN/test_32x32.mat")
    return train, test


class ClassificationConv():
    def __init__(self):
        super().__init__()

    def initialize(self, in_shape):
        self.network = keras.Sequential([
            keras.Input(shape=in_shape),
            keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                             loss=keras.losses.CategoricalCrossentropy, metrics=['Accuracy'])

    def train(self, train, label, val_tr, val_lb, epochs):
        return self.network.fit(train, label, batch_size=128, epochs=epochs, validation_data=(val_tr, val_lb))

    def pred(self, train):
        return self.network.predict(train)


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
    # Unable to allocate 168 KiB for array with Shape 32 32 21 and dtype float64, but dataset reduction fixes this
    return end


def classification_conv_test():  # 94.7% ACC after two Networks with TF between them with all Data.
    # --> TF and vector builder doesn't change anything
    print('conv_test')
    keras.backend.clear_session()
    z1 = time.perf_counter()
    name = 'Classification_Big_Net'
    a, b, c, d = mnist_loader()
    e, f, g, h = svhn_loader()
    ls = []
    # todo: Change the Epoch Number from 1 to 100 at free will
    epochs = 10  # Epochenanzahl pro iteration
    model_1 = ClassificationConv()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = build_vec_conv(e, pred)
    val_aug_vec = build_vec_conv(g, val_pred)
    x = 11
    # todo: Check the iteration number and take it up to 10 or 50: range number is allowed to change
    for i in range(1):  # Netzanzahl, die gelernt werden
        model = ClassificationConv()
        model.initialize((32, 32, x))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        # todo: My Computer needs the break here, due to memory errors!
        # break  # Evtl. nötiger Breakpoint, wegen Memory-Problemen.
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = build_vec_conv(augmented_vector, pred)
        x += 10

    z2 = time.perf_counter()
    class_mult_plots(add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


def add_epoch_counter_to_df(df):
    sort_list = []
    i = 0
    while i < len(df):
        sort_list.append(i)
        i += 1
    df['epochs'] = sort_list
    return df


def class_mult_plots(df, epochs, samples, time, name):
    plt.plot(df['epochs'], df['Accuracy'], 'y', label='Training Accuracy')
    plt.plot(df['epochs'], df['val_Accuracy'], 'r', label='Validation Accuracy')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    text_3 = ('Time: ' + str(time))
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, 0.8, tex)
    plt.legend()
    plt.show()


classification_conv_test()
