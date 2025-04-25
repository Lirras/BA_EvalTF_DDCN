# A Strange Network which adds for TF a new Layer before the Old Cascade Network.

import torch

from casunit import CascadeNetwork
from main import mnist_data_loader
from main import svhn_data_loader
from main import workflow
# from main import epochs
from main import test_new_layer


def inversive_cascade_tf():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = 'cpu'

    model = CascadeNetwork().to(device)

    epoch = 5
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    test_data, valid_data = mnist_data_loader(True, 100)

    # Start of Cascade Network
    model.add_layer(torch.nn.Identity())  # Identity is needed for changes at the Input shapes, due to the different
    # Datasets
    model.add_layer((-1, 784), 'reshape')
    model.add_layer(torch.nn.Linear(784, 784, device=device))
    workflow(model, epoch, device, test_data, valid_data)

    model.add_layer((100, 1, 28, 28), 'reshape')
    model.add_layer(torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu')
    # stride niedrig: local Features
    # stride hoch: global Features + weniger Overfitting, bessere Effizienz

    model.add_layer((-1, 784), 'reshape')
    workflow(model, epoch, device, test_data, valid_data)

    model.add_layer(torch.nn.Linear(784, 10, device=device))
    workflow(model, epoch, device, test_data, valid_data)

    new_test_data, new_valid_data = svhn_data_loader(100)

    # This is the Downscaling for in, out, kernel
    model.list_of_layers[0] = [torch.nn.Conv2d(3, 1, 5, stride=1, padding=0, padding_mode='zeros',
                                               device=device), 'relu']
    # Layer Preparation of first Layer
    for i in model.list_of_layers[0][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True

    # Training and Testing for the Whole Network with only first Layer Changes enabled
    # epochs(model, epoch, device, new_test_data, new_valid_data)
    own_epochs(model, epoch, new_test_data, new_valid_data, device)

    # Freezing of first Layer
    for i in model.list_of_layers[0][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = False


def own_epochs(model, epochs, dataset, valid, device):
    optimizer = torch.optim.SGD(model.list_of_layers[0][0].parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()

    i = 0
    while i <= epochs:
        train(model, dataset, optimizer, crit, device)
        test_new_layer(valid, model, device, crit)
        i += 1


def train(model, dataset, optimizer, crit, device):
    model.to(device)
    counter = 1
    print(len(dataset))
    for data, label in dataset:
        optimizer.zero_grad()
        # Data through the whole Network
        output = model.list_of_layers[0][0](data.to(device))  # New Layer training
        output = torch.reshape(output, (-1, 784))
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        counter += 1
        if counter == len(dataset):
            print(data.shape)
            break
    print(counter)


inversive_cascade_tf()
