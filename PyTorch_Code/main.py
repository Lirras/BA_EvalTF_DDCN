# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 06.05.2025
# Version: 0.0.011

from casunit import CascadeNetwork
import data_loader
import workflow as wf
import torch


# A Linear Layer has only Sense as One Layer and
    # with few epochs. Kinda 1-5
# todo: make the correct Shape of outputs at here with reshaping for output: 1, size with label: 1-Dim
# todo: This causes Error at first layer. (at Reshape) -> I need b, c, h, w with 100 1 28 28 not 100 3 32 32
# todo: Why is Conv so much worse than Linear?


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Strg+F8 to toggle the breakpoint.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CascadeNetwork().to(device)

    # scipy.datasets.download_all()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # device = 'cpu'

    epoch = 5
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    mnist_train, mnist_val = data_loader.mnist_data_loader(True, 100)
    # Load the SVHN Dataset
    svhn_train, svhn_val = data_loader.svhn_data_loader(100)  # with Last Data cutted.

    # todo: dropout + normalize Layer einfügen und verstehen, was letztere sind.
    # todo: Approach über torch_geometric machen -> Dataset apply -> I dont do this
    # todo: Maybe an Approach über Keras? -> Yes, of course!
    # todo: Digraph test

    # model.add_layer(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros').to(device), 'relu')
    model.add_layer(1, 'flatten')
    model.add_layer(torch.nn.Linear(1024, 10).to(device))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)
    
    # model.add_layer(1, 'batch_norm', [torch.zeros(32), torch.ones(32)])
    '''model.add_layer(1, 'flatten')
    model.add_layer(torch.nn.Linear(1024, 1024))
    wf.workflow(model, epoch, device, mnist_train, mnist_val)
    # 2Linear is better with 20.6% than the svhn_net alone

    model.add_layer(1, 'unflatten', (32, 32))
    model.add_layer(1, 'unsqueeze')
    # model.add_layer(torch.nn.Linear(1024, 1024))
    model.add_layer(torch.nn.Conv2d(1, 8, 3, stride=1, padding=1, padding_mode='zeros'), 'relu')
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
    # 17.0%
    model.add_layer(torch.nn.Linear(2048, 2048))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)'''
    # 20.8%
    # wf.workflow(model, epoch, device, svhn_train, svhn_val)

    # Reihenfolge von https://github.com/pitsios-s/SVHN/blob/master/src/cnn.py
    # norm, conv1(relu), maxpool1, conv2(relu), maxpool2, conv3(relu), maxpool3, FC(relu), Dropout
    '''model.add_layer(1, 'flatten')
    model.add_layer(torch.nn.Linear(1024, 1024).to(device))
    #wf.workflow(model, epoch, device, mnist_train, mnist_val)
    model.add_layer(1, 'unflatten', (32, 32))
    model.add_layer(1, 'unsqueeze')
    # 89.2%/91.1%/88.3%
    model.add_layer(1, 'batch_norm', [torch.zeros(1), torch.ones(1)], device)
    model.add_layer(torch.nn.Conv2d(1, 16, 3, stride=1, padding=1, padding_mode='zeros').to(device), 'relu')
    model.add_layer(2, 'max_pooling')
    model.add_layer(1, 'flatten')
    #wf.workflow(model, epoch, device, mnist_train, mnist_val)
    # 2.8%/16.5%/17.0%/16.3%
    model.list_of_layers[-1] = [torch.nn.Conv2d(16, 64, 3, stride=1, padding=1, padding_mode='zeros').to(device), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(1, 'flatten')
    #wf.workflow(model, epoch, device, mnist_train, mnist_val)
    # 10.0%/19.0%/6.7%/18.8%
    model.list_of_layers[-1] = [torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='zeros').to(device), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
    # 15.5%/9.6%/6.7%/9.0%
    model.add_layer(torch.nn.Linear(4096, 1024).to(device), 'relu')
    model.add_layer(torch.nn.Dropout(0.8).to(device), 'dropout')
    #wf.workflow(model, epoch, device, mnist_train, mnist_val)
    # 6.7%/6.7%/9.4%/8.6%
    model.add_layer(torch.nn.Linear(1024, 10).to(device))
    wf.workflow(model, epoch, device, svhn_train, svhn_val)'''
    # 19.6%/19.6% egal, ob das dritte Convlayer dabei ist oder nicht. Aber kein Vergleich zu ohne TF.
    # --> 19.6% bei egal was -> Das liegt an den Daten, die vom Dataloader kommen, aber mnist ist schlechter mit keras!
    # 87.5/17.0/14.7/8.3/19.6 L1M,C1S,C2S,L2S,L3S, epochs=1
    # 2.7/5.3/11.1/19.7 C1S,C2S,L1S,L2S, epochs=1
    # 16.3/16.6/15.7/8.1/19.6 L1S,C1S,C2S,L2S,L3S, epochs=1  -> TF changes Nothing here!
    # 88,7/21.4/19.0/10.9/19.2 L1M,C1M,C2M,L2M,L3S, epochs=1 all: 20.4

    '''model.add_layer(torch.nn.Conv2d(1, 8, 5, stride=1, padding=1, padding_mode='zeros'), 'relu')
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
    # 3.1%

    model.list_of_layers[-1] = [torch.nn.Conv2d(8, 16, 5, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
    # 12.0%

    model.list_of_layers[-1] = [torch.nn.Conv2d(16, 32, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)
    # 13.9%

    model.list_of_layers[-1] = [torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='zeros'), 'relu']
    model.add_layer(2, 'max_pooling')
    model.add_layer(torch.nn.Dropout(0.3), 'dropout')
    model.add_layer(1, 'flatten')
    wf.workflow(model, epoch, device, svhn_train, svhn_val)'''
    # 17.9%

    # All: ohne Norm
    # 2.7% ohne relu
    # 3.4% all
    # 2.9% ohne Dropout
    # 2.9% ohne Dropout, ohne relu
    # 3.0% ohne Dropout, ohne relu, ohne pooling
    # 2.8% ohne relu, ohne pooling
    # 3.0% ohne Dropout, ohne pooling
    # 2.9% ohne pooling

    # model.add_layer(torch.nn.Conv2d(8, 16, 5, stride=1, padding=1, padding_mode='zeros'))

    '''model.add_layer(1, 'flatten')
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
    wf.workflow(model, epoch, device, svhn_train, svhn_val)'''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
