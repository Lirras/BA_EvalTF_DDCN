import numpy as np
from sklearn.preprocessing import LabelBinarizer
from PyTorch_Code import data_loader


def crutch_returned():
    lb = LabelBinarizer()

    a, b, c, d, e, f, g, h = loader_crutch()  # Batch_size at data_loader is the size of the corresponded dataset
    b = lb.fit_transform(b)
    d = lb.fit_transform(d)
    f = lb.fit_transform(f)
    h = lb.fit_transform(h)
    b = b.astype('int64')
    d = d.astype('int64')
    f = f.astype('int64')
    h = h.astype('int64')
    a = a.astype('float')
    c = c.astype('float')
    e = e.astype('float')
    g = g.astype('float')

    a = np.moveaxis(a, 1, 3)  # B, C, H, W -> B, H, W ,C
    c = np.moveaxis(c, 1, 3)
    e = np.moveaxis(e, 1, 3)
    g = np.moveaxis(g, 1, 3)
    print(a.shape)
    print(c.shape)
    print(e.shape)
    print(g.shape)

    # With this Crutch:
    # 10 % ACC by MNIST
    # 19.6% ACC by SVHN


def loader_crutch():
    mnist_tr, mnist_va = data_loader.mnist_data_loader(True, 1)
    svhn_tr, svhn_va = data_loader.svhn_data_loader(1)
    mnist_tr_dat, mnist_tr_lb = loader_crutch_backend(mnist_tr)
    mnist_va_dat, mnist_va_lb = loader_crutch_backend(mnist_va)
    svhn_tr_dat, svhn_tr_lb = loader_crutch_backend(svhn_tr)
    svhn_va_dat, svhn_va_lb = loader_crutch_backend(svhn_va)
    return mnist_tr_dat, mnist_tr_lb, mnist_va_dat, mnist_va_lb, svhn_tr_dat, svhn_tr_lb, svhn_va_dat, svhn_va_lb


def loader_crutch_backend(dataloader):
    data = next(iter(dataloader))[0].numpy()
    label = next(iter(dataloader))[1].numpy()
    return data, label
