import workflow as wf
import torch
import data_loader as dl
from casunit import CascadeNetwork


def ConvMidTF():
    model = CascadeNetwork()
    epoch = 1
    device = 'cpu'  # or 'cuda' if possible
    mnist_train, mnist_val = dl.mnist_data_loader(True, 100)
    svhn_train, svhn_val = dl.svhn_data_loader(100)

    model.add_layer(1, 'flatten')
    model.add_layer(torch.nn.Linear(1024, 1024))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(1, 'unflatten', (32, 32))
    model.add_layer(1, 'unsqueeze')
    model.add_layer(torch.nn.Conv2d(1, 8, 5, stride=1, padding=1, padding_mode='zeros'), 'relu')  # 100 8 28 28
    model.add_layer(2, 'max_pooling')  # 100 8 14 14
    model.add_layer(0, 'batch_norm')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)  # 1800

    # model.list_of_layers[-1] = [0, 'batch_norm']
    # model.add_layer(1, 'flatten')
    # wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(8, 32, 5, stride=1, padding=2, padding_mode='zeros'), 'relu']
    # model.add_layer(torch.nn.Conv2d(8, 32, 5, stride=1, padding=2, padding_mode='zeros'), 'relu')  # 100 32 12 12
    model.add_layer(2, 'max_pooling')  # 100 32 6 6
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)  # 1568

    model.list_of_layers[-1] = [torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    # model.add_layer(torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='zeros'), 'relu')
    model.add_layer(2, 'max_pooling')  # 100 64 3 3
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)  # 576

    model.list_of_layers[-1] = [torch.nn.Conv2d(64, 100, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')  # 100 100 1 1
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)  # 100

    model.add_layer(torch.nn.Linear(100, 100))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    model.add_layer(torch.nn.Linear(100, 100))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    model.add_layer(torch.nn.Linear(100, 100))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    # model.add_layer(torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu')
    # model.add_layer((-1, 1024), 'reshape')
    # model.add_layer(torch.nn.Linear(1024, 1024))
    # wf.workflow(model, epoch, device, mnist_train, mnist_val)
    # model.add_layer(torch.nn.Linear(1024, 1024))
    # wf.workflow(model, epoch, device, svhn_train, svhn_val)
