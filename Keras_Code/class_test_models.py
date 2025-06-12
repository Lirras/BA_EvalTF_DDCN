import time
import pandas
import class_units
import plotting
import keras_data_loader
import keras_regressoion_lib as krl
import keras_cascade_lib as kcl


# todo: Metrik bauen für Anzahl Epochen pro Layer
# todo: Ergebnisse vorher in ein Dokument hinetereinander, damit ich die nicht suchen brauch.

# todo: Was wurde durchgeführt, wie und warum
# todo: Setup aufschreiben
# todo: Weniger Hidden Layer
# todo: lil Conv mit vectorbuild von class Dense kombinieren


def regression_test():
    krl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Regression_one_Layer'
    # todo: Check how big is the Data now
    a, b, c, d, bost_test_data, bost_test_target = keras_data_loader.boston_loader()
    e, f, g, h, cali_test_data, cali_test_target = keras_data_loader.california_loader()
    model_1 = class_units.Regression()
    x = 3
    ls = []
    test_plot = []
    model_1.initialize((x,))
    hist = model_1.train(a, b, c, d, 5)
    ls.append(pandas.DataFrame.from_dict(hist.history))

    bost_pred = model_1.pred(a)
    bost_val_pred = model_1.pred(c)
    bost_augmented_vector = krl.build_2nd_in_same(a, bost_pred)
    bost_val_aug_vec = krl.build_2nd_in_same(c, bost_val_pred)

    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = krl.build_2nd_in_same(e, pred)
    val_aug_vec = krl.build_2nd_in_same(g, val_pred)

    test_pred = model_1.pred(cali_test_data)
    test_plot.append(krl.build_mae_test(test_pred, cali_test_target))
    test_aug = krl.build_2nd_in_same(cali_test_data, test_pred)

    x += 1

    # boston
    for i in range(30):
        print(i)
        model = class_units.Regression()
        model.initialize((x,))
        hist = model.train(bost_augmented_vector, b, bost_val_aug_vec, d, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))

        # boston
        bost_pred = model.pred(bost_augmented_vector)
        bost_augmented_vector = krl.build_2nd_in_same(bost_augmented_vector, bost_pred)
        bost_val = model.pred(bost_val_aug_vec)
        bost_val_aug_vec = krl.build_2nd_in_same(bost_val_aug_vec, bost_val)

        # cali
        pred = model.pred(augmented_vector)
        augmented_vector = krl.build_2nd_in_same(augmented_vector, pred)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = krl.build_2nd_in_same(val_aug_vec, val_pred)

        # test
        test_pred = model.pred(test_aug)
        test_plot.append(krl.build_mae_test(test_pred, cali_test_target))
        test_aug = krl.build_2nd_in_same(test_aug, test_pred)
        x += 1

    # california
    for i in range(30):
        print(i)
        model = class_units.Regression()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        augmented_vector = krl.build_2nd_in_same(augmented_vector, pred)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = krl.build_2nd_in_same(val_aug_vec, val_pred)

        test_pred = model.pred(test_aug)
        test_plot.append(krl.build_mae_test(test_pred, cali_test_target))
        test_aug = krl.build_2nd_in_same(test_aug, test_pred)
        x += 1

    z2 = time.perf_counter()
    plotting.regr_test_plot(plotting.add_epoch_counter_to_df(pandas.DataFrame({'mae': test_plot})), epochs, len(cali_test_data), round(z2-z1), name)
    plotting.multiple_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2 - z1:0.2f} sec')


# These one is at mnist_long
def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()
    ls = []
    model_1 = class_units.Classification()
    # todo: model needs Flatten Layer
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
        print(i)
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


# Beware! This one needs a very big RAM
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
        print(i)
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


# Beware! This one needs a very big RAM!
def lil_conv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 1
    name = 'little_conv'
    a, b, c, d, test_dat_mnist, test_lb_mnist = keras_data_loader.mnist_loader()
    e, f, g, h, test_dat_svhn, test_lb_svhn = keras_data_loader.svhn_loader()  # full: 1/less: 10/ Very less: More
    ls = []
    test_plot_list = []
    model_1 = class_units.LittleConv()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    test_plot_list.append(plotting.preds_for_plots(model_1.pred(test_dat_mnist), test_lb_mnist))
    test_aug = kcl.build_vec_conv(test_dat_svhn, model_1.pred(test_dat_svhn))  # Testset is to Big
    augmented_vector = kcl.build_vec_conv(e, pred)
    val_aug_vec = kcl.build_vec_conv(g, val_pred)
    x = 11
    for i in range(1):  # Why is the amount of iterations insignificant?
        print(i)
        model = class_units.LittleConv()
        model.initialize((32, 32, x))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)

        test_plot_list.append(plotting.preds_for_plots(model.pred(test_aug), test_lb_svhn))
        test_aug = kcl.build_vec_conv(test_aug, model.pred(test_aug))
        val_aug_vec = kcl.build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_conv(augmented_vector, pred)
        x += 10
    z2 = time.perf_counter()

    plotting.class_acc_only(plotting.add_epoch_counter_to_df(pandas.DataFrame(test_plot_list)), epochs, len(e), round(z2-z1), name)

    plotting.class_mult_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


def class_Dense():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Class_Dense'
    a, b, c, d, test_dat_mnist, test_lab_mnist = keras_data_loader.mnist_loader()
    e, f, g, h, test_dat_svhn, test_lab_svhn = keras_data_loader.svhn_loader()
    a = a.reshape((len(a), 1024, 1))
    c = c.reshape((len(c), 1024, 1))
    e = e.reshape((len(e), 1024, 1))
    g = g.reshape((len(g), 1024, 1))
    # test_dat_mnist = test_dat_mnist.reshape((len(test_dat_mnist), 1024, 1))
    test_dat_svhn = test_dat_svhn.reshape((len(test_dat_svhn), 1024, 1))
    ls = []
    x = 1024

    model_1 = class_units.Class_Dense()
    model_1.initialize((x, 1))
    hist = model_1.train(a, b, c, d, 1)
    ls.append(pandas.DataFrame.from_dict(hist.history))

    # Source Data
    mpred = model_1.pred(a)
    mval_pred = model_1.pred(c)
    maugmented_vector = kcl.build_vec_for_dense(a, mpred)
    mval_aug_vec = kcl.build_vec_for_dense(c, mval_pred)

    # Target Data
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = kcl.build_vec_for_dense(e, pred)
    val_aug_vec = kcl.build_vec_for_dense(g, val_pred)

    # Test Data
    test_plot = []
    test_pred = model_1.pred(test_dat_svhn)
    test_plot.append(plotting.preds_for_plots(test_pred, test_lab_mnist))
    test_aug = kcl.build_vec_for_dense(test_dat_svhn, test_pred)
    x += 10

    # Source Networks
    for i in range(1):
        print(i)
        model = class_units.Class_Dense()
        model.initialize((x, 1))
        hist = model.train(maugmented_vector, b, mval_aug_vec, d, 1)
        ls.append(pandas.DataFrame.from_dict(hist.history))

        # Source Data
        mpred = model.pred(maugmented_vector)
        mval_pred = model.pred(mval_aug_vec)
        mval_aug_vec = kcl.build_vec_for_dense(mval_aug_vec, mval_pred)
        maugmented_vector = kcl.build_vec_for_dense(maugmented_vector, mpred)

        # Target Data
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_for_dense(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_for_dense(augmented_vector, pred)  # (1024, 1)

        # Test Data
        test_pred = model.pred(test_aug)
        test_aug = kcl.build_vec_for_dense(test_aug, test_pred)
        test_plot.append(plotting.preds_for_plots(test_pred, test_lab_svhn))

        x += 10

    # Target Networks
    for i in range(10):
        print(i)
        model = class_units.Class_Dense()
        model.initialize((x, 1))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))

        # Target Data
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_for_dense(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_for_dense(augmented_vector, pred)  # (1024, 1)

        # Test Data
        test_pred = model.pred(test_aug)
        test_aug = kcl.build_vec_for_dense(test_aug, test_pred)
        test_plot.append(plotting.preds_for_plots(test_pred, test_lab_svhn))

        x += 10

    z2 = time.perf_counter()
    plotting.class_networks(plotting.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_plot})), 2, len(e), round(z2 - z1), name)
    plotting.class_acc_only(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2 - z1), name)
    print(f'{z2 - z1:0.2f} sec')


# todo: only use here class_Dense and regression_test; The third Network is at mnist_long
class_Dense()
# lil_conv()
# classification_conv_test()
# classification_test()
# regression_test()
