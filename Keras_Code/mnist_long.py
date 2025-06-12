import time
import pandas
import keras_data_loader
import keras_cascade_lib as kcl
import class_units
import plotting


def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    epochs = 10
    name = 'Classification_one_Dense'
    a, b, c, d, mnist_test, mnist_test_lab = keras_data_loader.mnist_loader()
    e, f, g, h, svhn_test, svhn_test_lab = keras_data_loader.svhn_loader()
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
    test_ls_plot.append(plotting.preds_for_plots(test_pred, svhn_test_lab))
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
        test_ls_plot.append(plotting.preds_for_plots(test_pred, svhn_test_lab))
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
        test_ls_plot.append(plotting.preds_for_plots(test_pred, svhn_test_lab))
        test_aug = kcl.build_vec_dense_only(test_aug, test_pred, x, len(svhn_test))
        x += 10

    z2 = time.perf_counter()
    plotting.class_networks(plotting.add_epoch_counter_to_df(pandas.DataFrame({'accuracy': test_ls_plot})), epochs, len(svhn_test), round(z2-z1), name)
    plotting.class_acc_only(plotting.add_epoch_counter_to_df(pandas.concat(ls)), epochs, len(e), round(z2-z1), name)
    print(f'{z2-z1:0.2f} sec')


classification_test()
