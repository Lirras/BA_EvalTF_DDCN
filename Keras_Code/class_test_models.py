import time
import pandas
import class_units
import plotting
import keras_data_loader
import keras_regressoion_lib as krl
import keras_cascade_lib as kcl


# todo: Metrik bauen für Anzahl Epochen pro Layer
# todo: Ergebnisse vorher in ein Dokument hinetereinander, damit ich die nicht suchen brauch.

def regression_test():
    krl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Regression_one_Layer'
    a, b, c, d = keras_data_loader.boston_loader()
    e, f, g, h = keras_data_loader.california_loader()
    model_1 = class_units.Regression()
    x = 3
    ls = []
    model_1.initialize((x,))
    hist = model_1.train(a, b, c, d, 5)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = krl.build_2nd_in_same(e, pred)
    val_aug_vec = krl.build_2nd_in_same(g, val_pred)
    x += 1

    for i in range(100):
        model = class_units.Regression()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        augmented_vector = krl.build_2nd_in_same(augmented_vector, pred)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = krl.build_2nd_in_same(val_aug_vec, val_pred)
        x += 1

    z2 = time.perf_counter()
    plotting.multiple_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2 - z1:0.2f} sec')


def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()
    ls = []
    model_1 = class_units.Classification()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, 5)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    in_shape = len(e)
    augmented_vector = kcl.build_vec_dense_only(e, pred, 1024, in_shape)
    val_aug_vec = kcl.build_vec_dense_only(g, val_pred, 1024, len(g))
    x = 1034
    for i in range(100):
        model = class_units.Classification()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_dense_only(val_aug_vec, val_pred, x, len(g))
        augmented_vector = kcl.build_vec_dense_only(augmented_vector, pred, x, in_shape)
        x += 10

    z2 = time.perf_counter()
    plotting.class_acc_only(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2-z1:0.2f} sec')


def classification_conv_test():  # 94.7% ACC after two Networks with TF between them with all Data.
    # --> TF and vector builder doesn't change anything
    print('conv_test')
    kcl.clear()
    z1 = time.perf_counter()
    name = 'Classification_Big_Net'
    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()
    ls = []
    epochs = 10
    model_1 = class_units.ClassificationConv()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = kcl.build_vec_conv(e, pred)
    val_aug_vec = kcl.build_vec_conv(g, val_pred)
    x = 11
    for i in range(50):  # Why is the amount of iterations insignificant?
        model = class_units.ClassificationConv()
        model.initialize((32, 32, x))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        # break
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_conv(augmented_vector, pred)
        x += 10

    z2 = time.perf_counter()
    plotting.class_mult_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


def lil_conv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'little_conv'
    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()  # full: 1/less: 10/ Very less: More
    ls = []
    model_1 = class_units.LittleConv()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = kcl.build_vec_conv(e, pred)
    val_aug_vec = kcl.build_vec_conv(g, val_pred)
    x = 11
    for i in range(100):  # Why is the amount of iterations insignificant?
        model = class_units.LittleConv()
        model.initialize((32, 32, x))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_conv(augmented_vector, pred)
        x += 10

    z2 = time.perf_counter()
    plotting.class_mult_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


lil_conv()
# classification_conv_test()
# classification_test()
# regression_test()
