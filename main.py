# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 28.04.2025
# Version: 0.0.007

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

    epoch = 5
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # for data, label in mnist_train:
        # for item in label:
        #     if item == 10:
        #         print('Zehn!')
        # break
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)  # with Last Data cutted.
    some_data = None
    for data, label in svhn_train:
        # todo: Something like that to display the Images of the Dataset.
        # for item in label:
        #     if item == 10:
        #         print('problem!')
        # x = data.squeeze()
        # numpy_array = x.numpy()
        # PIL.Image.open(plt.imshow(numpy_array))
        # plt.show()
        some_data = data
        print(f"label: {label.shape}")
        print(f"data: {data.shape}")
        break

    # todo: dropout + normalize Layer einfügen und verstehen, was letztere sind.
    # todo: Approach über torch_geometric machen

    model.add_layer(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros'), 'relu')
    model.add_layer(9, 'batch_norm', [torch.zeros(32), torch.ones(32)])
    model.forward(some_data)
    print(model.list_of_layers[-1][0])  # That is a Tensor!
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, svhn_train, svhn_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(1, 'batch_norm', [torch.zeros(64), torch.ones(64)])
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(1, 'batch_norm', [torch.zeros(128), torch.ones(128)])
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(128, 128, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.add_layer(torch.nn.Linear(2048, 128), 'relu')
    model.add_layer(torch.nn.Dropout(0.4), 'dropout')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)

    model.add_layer(torch.nn.Linear(128, 10), 'softmax')
    running_dataset = wf.workflow(model, epoch, device, running_dataset, svhn_val)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
