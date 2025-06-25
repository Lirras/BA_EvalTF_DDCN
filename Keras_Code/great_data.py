import time
import Keras_Code.DirectCascade.DenseTests as dens
import Keras_Code.DeepCascade.models_01 as mod1
import Keras_Code.DirectCascade.ConvClassTests as conv
import Keras_Code.libraries.plotting as pltt


def cmp():
    name = 'Conv_MaxPool'
    # epochs = 10
    z1 = time.perf_counter()
    tr_df0, ts_df0, leng0 = mod1.cascade_network(0.99)
    z2 = time.perf_counter()
    tr_df1, ts_df1, leng1 = mod1.cascade_network(0.9)
    z3 = time.perf_counter()
    tr_df2, ts_df2, leng2 = mod1.cascade_network(0.7)
    z4 = time.perf_counter()
    tr_df3, ts_df3, leng3 = mod1.cascade_network(0.5)
    z5 = time.perf_counter()
    tr_df4, ts_df4, leng4 = mod1.cascade_network(0.3)
    z6 = time.perf_counter()
    tr_dfs = [tr_df0, tr_df1, tr_df2, tr_df3, tr_df4]
    ts_dfs = [ts_df0, ts_df1, ts_df2, ts_df3, ts_df4]
    samples = [leng0, leng1, leng2, leng3, leng4]
    timings = [round(z2-z1), round(z3-z2), round(z4-z3), round(z5-z4), round(z6-z5)]
    pltt.great_data_class_train(tr_dfs, name, samples, timings)
    pltt.great_data_class_test(ts_dfs, name, samples, timings)


def cod():
    name = 'ClassOneDense'
    # epochs = 10
    z1 = time.perf_counter()
    tr_df0, ts_df0, leng0 = dens.classification_test(0.99)
    z2 = time.perf_counter()
    tr_df1, ts_df1, leng1 = dens.classification_test(0.9)
    z3 = time.perf_counter()
    tr_df2, ts_df2, leng2 = dens.classification_test(0.7)
    z4 = time.perf_counter()
    tr_df3, ts_df3, leng3 = dens.classification_test(0.5)
    z5 = time.perf_counter()
    tr_df4, ts_df4, leng4 = dens.classification_test(0.3)
    z6 = time.perf_counter()
    tr_dfs = [tr_df0, tr_df1, tr_df2, tr_df3, tr_df4]
    ts_dfs = [ts_df0, ts_df1, ts_df2, ts_df3, ts_df4]
    samples = [leng0, leng1, leng2, leng3, leng4]
    timings = [round(z2-z1), round(z3-z2), round(z4-z3), round(z5-z4), round(z6-z5)]
    pltt.great_data_class_train_small(tr_dfs, name, samples, timings)
    pltt.great_data_class_test(ts_dfs, name, samples, timings)


def onedconv():
    name = '1DConv'
    # epochs = 10
    z1 = time.perf_counter()
    tr_df0, ts_df0, leng0 = conv.OneDLilConv(0.99)
    z2 = time.perf_counter()
    tr_df1, ts_df1, leng1 = conv.OneDLilConv(0.9)
    z3 = time.perf_counter()
    tr_df2, ts_df2, leng2 = conv.OneDLilConv(0.7)
    z4 = time.perf_counter()
    tr_df3, ts_df3, leng3 = conv.OneDLilConv(0.5)
    z5 = time.perf_counter()
    tr_df4, ts_df4, leng4 = conv.OneDLilConv(0.3)
    z6 = time.perf_counter()
    tr_dfs = [tr_df0, tr_df1, tr_df2, tr_df3, tr_df4]
    ts_dfs = [ts_df0, ts_df1, ts_df2, ts_df3, ts_df4]
    samples = [leng0, leng1, leng2, leng3, leng4]
    timings = [round(z2-z1), round(z3-z2), round(z4-z3), round(z5-z4), round(z6-z5)]
    pltt.great_data_class_train(tr_dfs, name, samples, timings)
    pltt.great_data_class_test(ts_dfs, name, samples, timings)


cod()  # Small
# cmp()  # Big
# onedconv()  # Big
