import time
import pandas
import class_units
import plotting
import keras_data_loader
import keras_regressoion_lib as krl
import keras_cascade_lib as kcl


# todo: In den Plots Epochen bzgl. Layer angeben, Wieviele Datensamples sind es gerade?
# todo: Metrik bauen für Anzahl Epochen pro Layer
# todo: Erst nachdem direct Cascade läuft
# todo: Labels oder irgendetwas ins Konzept.tex einbauen, damit man weiß, welche Graphik wozu gehört
# todo: Ergebnisse vorher in ein Dokument hinetereinander, damit ich die nicht suchen brauch.

def regression_test():
    krl.clear()
    z1 = time.perf_counter()
    a, b, c, d = keras_data_loader.boston_loader()
    e, f, g, h = keras_data_loader.california_loader()
    model_1 = class_units.Regression()
    x = 3
    ls = []
    model_1.initialize((x,))
    hist = model_1.train(a, b, c, d)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    augmented_vector = krl.build_2nd_in_same(e, pred)
    val_aug_vec = krl.build_2nd_in_same(g, val_pred)
    x += 1

    for i in range(20):
        model = class_units.Regression()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        augmented_vector = krl.build_2nd_in_same(augmented_vector, pred)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = krl.build_2nd_in_same(val_aug_vec, val_pred)
        x += 1

    plotting.multiple_plots(plotting.add_epoch_counter_to_df(pandas.concat(ls)))
    z2 = time.perf_counter()
    print(f'{z2 - z1:0.2f} sec')


def classification_test():
    kcl.clear()
    z1 = time.perf_counter()
    a, b, c, d = keras_data_loader.mnist_loader()
    e, f, g, h = keras_data_loader.svhn_loader()
    ls = []
    model_1 = class_units.Classification()
    model_1.initialize((32, 32, 1))
    hist = model_1.train(a, b, c, d)
    ls.append(pandas.DataFrame.from_dict(hist.history))
    pred = model_1.pred(e)
    val_pred = model_1.pred(g)
    in_shape = len(e)
    augmented_vector = kcl.build_vec_dense_only(e, pred, 1024, in_shape)
    val_aug_vec = kcl.build_vec_dense_only(g, val_pred, 1024, len(g))
    x = 1034
    for i in range(20):
        model = class_units.Classification()
        model.initialize((x,))
        hist = model.train(augmented_vector, f, val_aug_vec, h)
        ls.append(pandas.DataFrame.from_dict(hist.history))
        pred = model.pred(augmented_vector)
        val_pred = model.pred(val_aug_vec)
        val_aug_vec = kcl.build_vec_dense_only(val_aug_vec, val_pred, x, len(g))
        augmented_vector = kcl.build_vec_dense_only(augmented_vector, pred, x, in_shape)
        x += 10

    plotting.class_acc_only(plotting.add_epoch_counter_to_df(pandas.concat(ls)))

    z2 = time.perf_counter()
    print(f'{z2-z1:0.2f} sec')


# classification_test()
regression_test()
