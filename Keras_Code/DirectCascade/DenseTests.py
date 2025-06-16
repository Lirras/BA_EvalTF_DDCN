import time
import pandas
import class_units
import Keras_Code.libraries.plotting as pltt
import Keras_Code.libraries.keras_data_loader as dat_loader
import Keras_Code.libraries.keras_regressoion_lib as krl
import Keras_Code.libraries.keras_cascade_lib as kcl


# todo: Metrik bauen für Anzahl Epochen pro Layer

# todo: Ergebnisse vorher in ein Dokument hinetereinander, damit ich die nicht suchen brauch.

# todo: Was wurde durchgeführt, wie und warum
# todo: Setup aufschreiben


def regression_test():
    krl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Regression_one_Layer'
    # todo: Check how big is the Data now
    print('Bost:')
    a, b, c, d, bost_test_data, bost_test_target = dat_loader.boston_loader()
    print('Cali:')
    e, f, g, h, cali_test_data, cali_test_target = dat_loader.california_loader()
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
    for i in range(10):
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
    pltt.regr_test_plot(pltt.add_epoch_counter_to_df(pandas.DataFrame({'mae': test_plot})), epochs, len(cali_test_data), round(z2-z1), name)
    pltt.multiple_plots(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2 - z1:0.2f} sec')


def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    a, b, c, d, mnist_test, mnist_test_lab = dat_loader.mnist_loader()
    e, f, g, h, svhn_test, svhn_test_lab = dat_loader.svhn_loader()
    x = 1024
    a = a.reshape(len(a), x)
    c = c.reshape(len(c), x)
    # mnist_test = mnist_test.reshape(len(mnist_test), x)
    e = e.reshape(len(e), x)
    g = g.reshape(len(g), x)
    svhn_test = svhn_test.reshape(len(svhn_test), x)
    ls = []
    test_ls_plot = []

    model_before = class_units.Classification()
    model_before.initialize((x,))
    hist = model_before.train(a, b, c, d, epochs)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_before.pred(a)
    val_pred = model_before.pred(c)
    mnist_augmented_vector = kcl.build_vec_dense_only(a, pred, x, len(a))
    mnist_val_aug_vec = kcl.build_vec_dense_only(c, val_pred, x, len(c))

    svhn_pred = model_before.pred(e)
    svhn_val_pred = model_before.pred(g)
    augmented_vector = kcl.build_vec_dense_only(e, svhn_pred, x, len(e))
    val_aug_vec = kcl.build_vec_dense_only(g, svhn_val_pred, x, len(g))

    test_pred = model_before.pred(svhn_test)
    test_ls_plot.append(pltt.preds_for_plots(test_pred, svhn_test_lab))
    test_aug = kcl.build_vec_dense_only(svhn_test, test_pred, x, len(svhn_test))
    x += 10

    # MNIST Iterations:
    for i in range(10):
        print(i)
        model_1 = class_units.Classification()
        model_1.initialize((x,))

        hist = model_1.train(mnist_augmented_vector, b, mnist_val_aug_vec, d, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))

        print(mnist_augmented_vector.shape)  # 60000 1034
        pred = model_1.pred(mnist_augmented_vector)
        val_pred = model_1.pred(mnist_val_aug_vec)
        mnist_augmented_vector = kcl.build_vec_dense_only(mnist_augmented_vector, pred, x, len(a))
        mnist_val_aug_vec = kcl.build_vec_dense_only(mnist_val_aug_vec, val_pred, x, len(c))

        print(augmented_vector.shape)  # 732 1034
        svhn_pred = model_1.pred(augmented_vector)
        svhn_val_pred = model_1.pred(val_aug_vec)
        augmented_vector = kcl.build_vec_dense_only(augmented_vector, svhn_pred, x, len(e))
        val_aug_vec = kcl.build_vec_dense_only(val_aug_vec, svhn_val_pred, x, len(g))

        test_pred = model_1.pred(test_aug)
        test_ls_plot.append(pltt.preds_for_plots(test_pred, svhn_test_lab))
        test_aug = kcl.build_vec_dense_only(test_aug, test_pred, x, len(svhn_test))
        x += 10

    in_shape = len(e)
    for i in range(50):
        print(i)
        model = class_units.Classification()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h, epochs)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_dense_only(val_aug_vec, val_pred, x, len(g))
        augmented_vector = kcl.build_vec_dense_only(augmented_vector, pred, x, in_shape)

        test_pred = model.pred(test_aug)
        test_ls_plot.append(pltt.preds_for_plots(test_pred, svhn_test_lab))
        test_aug = kcl.build_vec_dense_only(test_aug, test_pred, x, len(svhn_test))
        x += 10

    z2 = time.perf_counter()
    pltt.class_networks(pltt.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_ls_plot})), epochs, len(svhn_test), round(z2-z1), name)
    pltt.class_acc_only(pltt.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2-z1:0.2f} sec')


# todo: only use here class_Dense and regression_test; The third Network is at mnist_long

classification_test()
# regression_test()
