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

    epoch = 20
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)  # with Last Data cutted.

    # todo: dropout + normalize Layer einfügen und verstehen, was letztere sind.
    # todo: Approach über torch_geometric machen -> Dataset apply
    # todo: Maybe an Approach über Keras?
    # todo: Digraph test
    
    # model.add_layer(1, 'batch_norm', [torch.zeros(32), torch.ones(32)])
    model.add_layer(1, 'flatten')
    model.add_layer(torch.nn.Linear(1024, 2048), 'relu')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    print("2nd Layer")
    model.add_layer(torch.nn.Linear(2048, 1024), 'relu')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    print("3rd Layer")
    model.add_layer(torch.nn.Linear(1024, 400), 'relu')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    print("TF")
    model.add_layer(torch.nn.Linear(400, 300), 'relu')
    model.add_layer(torch.nn.Dropout(0.4), 'dropout')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    print("5th Layer")
    model.add_layer(torch.nn.Linear(300, 200), 'relu')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)

    print("6th Layer")
    model.add_layer(torch.nn.Linear(200, 10), 'relu')
    model.add_layer(torch.nn.Dropout(0.4), 'dropout')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
