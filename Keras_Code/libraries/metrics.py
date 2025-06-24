import pandas
from sklearn.model_selection import cross_val_score
import sklearn.metrics as skmet


def acc_met(network, tr_dat, tr_lb, val_dat, val_lb, epochs):
    list_of_dfs = []
    for i in range(epochs):
        print('Epoch: ' + str(i))
        hist = network.fit(tr_dat, tr_lb, batch_size=128, epochs=1, validation_data=(val_dat, val_lb))
        df = pandas.DataFrame.from_dict(hist.history)
        list_of_dfs.append(df)
        acc = df['Accuracy'][0]
        val_acc = df['val_Accuracy'][0]
        if acc-val_acc > 0.1:
            break
    return pandas.concat(list_of_dfs)


def loss_met(network, tr_dat, tr_lb, val_dat, val_lb, epochs):
    list_of_dfs = []
    before_loss = (2 ** 30)
    for i in range(epochs):
        print('Epoch: ' + str(i))
        hist = network.fit(tr_dat, tr_lb, batch_size=128, epochs=1, validation_data=(val_dat, val_lb))
        df = pandas.DataFrame.from_dict(hist.history)
        if i != 0:
            before_loss = list_of_dfs[-1]['val_loss'][0]
        list_of_dfs.append(df)
        # loss = df['loss'][0]
        val_loss = df['val_loss'][0]
        if before_loss < val_loss:
            break
    return pandas.concat(list_of_dfs)


def mae_met(network, tr_dat, tr_lb, val_dat, val_lb, epochs):
    list_of_dfs = []
    before_loss = (2 ** 30)
    for i in range(epochs):
        print('Epoch: ' + str(i))
        hist = network.fit(tr_dat, tr_lb, batch_size=16, epochs=1, validation_data=(val_dat, val_lb))
        df = pandas.DataFrame.from_dict(hist.history)
        if i != 0:
            before_loss = list_of_dfs[-1]['val_mae'][0]  # todo: This cant be used at first
        list_of_dfs.append(df)
        val_mae = df['val_mae'][0]
        if before_loss < val_mae:
            break
    return pandas.concat(list_of_dfs)


def kfoldcross(network, tr_dat, tr_lab, epochs):
    mean_ls = []
    std_ls = []
    for i in range(epochs):
        out = cross_val_score(network, tr_dat, tr_lab, scoring=skmet.make_scorer(skmet.mean_squared_error), cv=5)
        mean_ls.append(out.mean())
        std_ls.append(out.std())
    mean = pandas.DataFrame({'mae': mean_ls})
    std = pandas.DataFrame({'std': std_ls})
    return mean, std
