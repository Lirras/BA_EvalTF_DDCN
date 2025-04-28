from casunit import CascadeNetwork
import torch
import data_loader
import workflow as wf


def ConLin3x3():

    model = CascadeNetwork()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = 'cpu'

    epoch = 5
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)

    model.add_layer(torch.nn.Conv2d(1, 4, 7, stride=1, padding=2, padding_mode='zeros', device=device), 'relu')
    model.add_layer(2, 'max_pooling')
    model.add_layer(1, 'flatten')  # 67600
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(4, 16, 5, stride=2, padding=1, padding_mode='zeros', device=device), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(1, 'flatten')  # 14400
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(16, 64, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu']
    model.add_layer(2, 'max_pooling')

    model.add_layer(1, 'flatten')  # 6400
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(64, 64))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(64, 64))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(64, 64))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(64, 64))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    model.add_layer(torch.nn.Linear(64, 64))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
