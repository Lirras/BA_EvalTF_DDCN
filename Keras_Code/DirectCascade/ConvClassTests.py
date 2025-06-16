import time
import pandas
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_cascade_lib as kcl
import class_units
import Keras_Code.libraries.plotting as pltt


def OneDLilConv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'OneDLilConv'
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

    model_1 = class_units.Class_Dense()
    model_1.initialize((x, 1))
    hist = model_1.train(a, b, c, d, epochs)
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
    for i in range(9):
        print(i)
        model = class_units.Class_Dense()
        model.initialize((x, 1))
        hist = model.train(maugmented_vector, b, mval_aug_vec, d, epochs)
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
        test_plot.append(pltt.preds_for_plots(test_pred, test_lab_svhn))

        x += 10

    z2 = time.perf_counter()
    pltt.class_networks(pltt.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_plot})), 2, len(e), round(z2 - z1), name)
    pltt.class_acc_only(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2 - z1), name)
    print(f'{z2 - z1:0.2f} sec')


# Beware! This one needs a very big RAM!
def lil_conv():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'little_conv'
    a, b, c, d, test_dat_mnist, test_lb_mnist = dat_loader.mnist_loader()
    e, f, g, h, test_dat_svhn, test_lb_svhn = dat_loader.svhn_loader()  # full: 1/less: 10/ Very less: More
    print(a.shape)
    print(c.shape)
    ls = []
    test_plot_list = []
    model_1 = class_units.LittleConv()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))

    # Source
    mpred = model_1.pred(a)
    mval_pred = model_1.pred(c)
    maugmented_vector = kcl.build_vec_conv(a, mpred)
    mval_aug_vec = kcl.build_vec_conv(c, mval_pred)

    # Target
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = kcl.build_vec_conv(e, pred)
    val_aug_vec = kcl.build_vec_conv(g, val_pred)

    # Test
    test_pred = model_1.pred(test_dat_svhn)
    test_plot_list.append(pltt.preds_for_plots(test_pred, test_lb_svhn))
    test_aug = kcl.build_vec_conv(test_dat_svhn, model_1.pred(test_dat_svhn))  # Testset is to Big

    x = 11

    # Source
    for i in range(9):
        print(i)
        model = class_units.LittleConv()
        model.initialize((32, 32, x))
        hist = model.train(maugmented_vector, b, mval_aug_vec, d, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))

        # Source
        mpred = model.pred(maugmented_vector)
        mval_pred = model.pred(mval_aug_vec)
        maugmented_vector = kcl.build_vec_conv(maugmented_vector, mpred)
        mval_aug_vec = kcl.build_vec_conv(mval_aug_vec, mval_pred)

        # Test
        test_pred = model.pred(test_aug)
        test_plot_list.append(pltt.preds_for_plots(test_pred, test_lb_svhn))
        test_aug = kcl.build_vec_conv(test_aug, test_pred)

        # Target
        val_pred = model.pred(val_aug_vec)
        pred = model.pred(augmented_vector)
        val_aug_vec = kcl.build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_conv(augmented_vector, pred)
        x += 10

    # Target
    for i in range(10):  # Why is the amount of iterations insignificant?
        print(i)
        model = class_units.LittleConv()
        model.initialize((32, 32, x))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)

        test_plot_list.append(pltt.preds_for_plots(model.pred(test_aug), test_lb_svhn))
        test_aug = kcl.build_vec_conv(test_aug, model.pred(test_aug))
        val_aug_vec = kcl.build_vec_conv(val_aug_vec, val_pred)
        augmented_vector = kcl.build_vec_conv(augmented_vector, pred)
        x += 10
    z2 = time.perf_counter()

    pltt.class_networks(pltt.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_plot_list})), epochs, len(e), round(z2-z1), name)
    pltt.class_mult_plots(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)

    print(f'{z2 - z1:0.2f} sec')


# OneDLilConv()
lil_conv()
