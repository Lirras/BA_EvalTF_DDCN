import Keras_Code.libraries.keras_cascade_lib as kcl
import time
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.DirectCascade.class_units as clun
import big_networks_class
import pandas
import Keras_Code.libraries.plotting as pltt


# These one is at mnist_long
def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()
    ls = []
    model_1 = clun.Classification()
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
        model = clun.Classification()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_dense_only(val_aug_vec, val_pred, x, len(g))
        augmented_vector = kcl.build_vec_dense_only(augmented_vector, pred, x, in_shape)
        x += 10

    z2 = time.perf_counter()
    pltt.class_acc_only(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2-z1:0.2f} sec')


# Beware! This one needs a very big RAM
def classification_conv_test():  # 94.7% ACC after two Networks with TF between them with all Data.
    # --> TF and vector builder doesn't change anything
    print('conv_test')
    kcl.clear()
    z1 = time.perf_counter()
    name = 'Classification_Big_Net'
    a, b, c, d = dat_loader.mnist_loader()
    e, f, g, h = dat_loader.svhn_loader()
    ls = []
    epochs = 10
    model_1 = big_networks_class.ClassificationConv()
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
        model = big_networks_class.ClassificationConv()
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
    pltt.class_mult_plots(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


# Big Network for Cascade
def class_Dense():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Class_Dense'
    a, b, c, d, test_dat_mnist, test_lab_mnist = dat_loader.mnist_loader()
    e, f, g, h, test_dat_svhn, test_lab_svhn = dat_loader.svhn_loader()
    a = a.reshape((len(a), 1024, 1))
    c = c.reshape((len(c), 1024, 1))
    e = e.reshape((len(e), 1024, 1))
    g = g.reshape((len(g), 1024, 1))
    # test_dat_mnist = test_dat_mnist.reshape((len(test_dat_mnist), 1024, 1))
    test_dat_svhn = test_dat_svhn.reshape((len(test_dat_svhn), 1024, 1))
    ls = []
    x = 1024

    model_1 = big_networks_class.Class_Dense()
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
    test_plot.append(pltt.preds_for_plots(test_pred, test_lab_mnist))
    test_aug = kcl.build_vec_for_dense(test_dat_svhn, test_pred)
    x += 10

    # Source Networks
    for i in range(1):
        print(i)
        model = big_networks_class.Class_Dense()
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
        test_plot.append(pltt.preds_for_plots(test_pred, test_lab_svhn))

        x += 10

    # Target Networks
    for i in range(10):
        print(i)
        model = big_networks_class.Class_Dense()
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
        test_plot.append(pltt.preds_for_plots(test_pred, test_lab_svhn))

        x += 10

    z2 = time.perf_counter()
    pltt.class_networks(pltt.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_plot})), 2, len(e), round(z2 - z1), name)
    pltt.class_acc_only(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2 - z1), name)
    print(f'{z2 - z1:0.2f} sec')

