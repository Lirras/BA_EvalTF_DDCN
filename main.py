# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 29.04.2025
# Version: 0.0.008

from casunit import CascadeNetwork
import data_loader
import workflow as wf
import torch
import train_of_Layers as tol
# import torch_geometric
# import scipy
# import torchvision
# import numpy
# import matplotlib.pyplot as plt
# import PIL.Image


# A Linear Layer has only Sense as One Layer and
    # with few epochs. Kinda 1-5
# todo: make the correct Shape of outputs at here with reshaping for output: 1, size with label: 1-Dim
# todo: This causes Error at first layer. (at Reshape) -> I need b, c, h, w with 100 1 28 28 not 100 3 32 32
# todo: Why is Conv so much worse than Linear?


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Strg+F8 to toggle the breakpoint.

    model = CascadeNetwork()

    # scipy.datasets.download_all()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = 'cpu'

    epoch = 5
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)  # with Last Data cutted.
    some_data = None
    for data, label in svhn_train:
        some_data = data
        break

    # Idee, das mit laufenden Daten zu machen, anstatt über eine Forward Method.

    # todo: dropout + normalize Layer einfügen und verstehen, was letztere sind.
    # todo: Approach über torch_geometric machen

    running_dataset = tol.conv2d(svhn_train, 1, 32, 3, 1, 1, 'zeros', 'relu')
    # todo: Auf neuen Datensatz ACC berechnung machen
    # ACC(running_dataset)

    # dataset = []
    # for data, label in svhn_train:
    #    output = torch.nn.functional.relu(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros')(data))
    #    dataset.append((output, label))

    # running_dataset = []
    # for data, label in dataset:
    #    output = torch.nn.functional.batch_norm(data, torch.zeros(32), torch.ones(32))
    #    running_dataset.append((output, label))

    # model.add_layer(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros'), 'relu')
    # Batch_norm is a Layer, not an activation
    # model.add_layer(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros'), 'batch_norm', [torch.zeros(32), torch.ones(32)])
    # model.add_layer(9, 'batch_norm', [torch.zeros(32), torch.ones(32)])
    # model.forward(some_data)
    # print(model.list_of_layers[-1][0])  # That is a Tensor!
    # todo: Vergleich mit Label, loss berechnen, update weights
    # model.add_layer(1, 'flatten')
    #  running_dataset = wf.workflow(model, epoch, device, svhn_train, svhn_val)


def ACC(dataset):
    length = len(dataset.dataset)
    batches = len(dataset)
    crit = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0
    with torch.no_grad():
        for data, label in dataset:
            loss += crit(data, label).item()
            correct += (data.argmax(1) == label).type(torch.float).sum().item()
    loss /= batches
    correct /= length
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
