# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 28.04.2025
# Version: 0.0.006

from casunit import CascadeNetwork
import data_loader
import workflow as wf
import torch
# import torchvision
# import numpy


# todo: With TF is here an error
    # todo: The Datasets are 40000, 10000, 73257, 26032 -> Batch_size=100 -> Size change at last iteration of Epoch
# A Linear Layer has only Sense as One Layer and
    # with few epochs. Kinda 1-5
# todo: make the correct Shape of outputs at here with reshaping for output: 1, size with label: 1-Dim
# todo: This causes Error at first layer. (at Reshape) -> I need b, c, h, w with 100 1 28 28 not 100 3 32 32
# todo: Why is Conv so much worse than Linear?


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Strg+F8 to toggle the breakpoint.

    model = CascadeNetwork()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = 'cpu'

    epoch = 1
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)  # Maybe Cut the Last Data?
    for data, label in svhn_train:
        print(f"data: {data.shape}")
        break

    # model.add_layer(torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu')
    model.add_layer((-1, 784), 'reshape')
    model.add_layer(torch.nn.Linear(784, 10))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)

    '''model.add_layer(torch.nn.Conv2d(1, 4, 7, stride=1, padding=2, padding_mode='zeros', device=device), 'relu')
    model.add_layer(2, 'max_pooling')
    model.add_layer(0, 'flatten')  # 67600
    # workflow(model, epoch, device, mnist_train, mnist_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(4, 16, 5, stride=2, padding=1, padding_mode='zeros', device=device), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(0, 'flatten')  # 14400
    # workflow(model, epoch, device, mnist_train, mnist_val)

    model.list_of_layers[-1] = [torch.nn.Conv2d(16, 64, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu']
    model.add_layer(2, 'max_pooling')

    model.add_layer(0, 'flatten')  # 6400
    # workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(6400, 1600))
    # workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(1600, 400))
    # workflow(model, epoch, device, mnist_train, mnist_val)

    model.add_layer(torch.nn.Linear(400, 100))
    workflow(model, epoch, device, mnist_train, mnist_val)
    workflow(model, epoch, device, svhn_train, svhn_val)'''


# This is from Fully Convolutional Neural Network Structure
# and Its Loss Function for Image Classification by QIUYU ZHU , (Member, IEEE), AND XUEWEN ZU
# This is disfunctional:
def softmax_loss(batch_size, output, label):
    return (1/batch_size)*sum(-math.log(math.exp(label)/sum(math.exp(output))))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
