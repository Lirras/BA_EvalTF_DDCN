# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 17.04.2025
# Version: 0.0.003

from casunit import CascadeNetwork
import torch
import torchvision
import numpy


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Strg+F8 to toggle the breakpoint.

    model = CascadeNetwork()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = 'cpu'

    epoch = 30
    # todo: hier eine Iteration in der alle Layer bzgl. des gerade Gegenwärtigen Losses hinzufügen.
    # todo: Problem: es dauert immer länger hier, je länger das Netz wird.
    input_size = 28*28
    model.add_layer((-1, 784), 'reshape')
    model.add_layer(torch.nn.Linear(input_size, 10).to(device))
    workflow(model, epoch, device)

    # model.add_layer(torch.nn.Linear(10, 10).to(device))
    # workflow(model, epoch, device)

    # model.add_layer(torch.nn.Linear(10, 10).to(device))
    # workflow(model, epoch, device)

    # model.add_layer(torch.nn.Linear(128, 10).to(device))
    # workflow(model, epoch, device)
    # model.add_layer(torch.nn.Conv2d(1, 1, 2).to(device), 'relu')
    # workflow(model, epoch, device)

    # model.add_layer(torch.nn.Conv2d(10, 10, 5).to(device), 'relu')
    # workflow(model, epoch, device)
    # sv_data = save_output(model, sv_data)

    # todo: Transfer-learning Wechsel zu dem nächsten Netzwerk, welches ein Cascade ist.

    '''train_dataset = data_loader(True, 64)
    test_dataset = data_loader(False, 64)

    for x, y in test_dataset:
        print(f"x: {x.shape}")  # 1, 1, 28, 28
        print(f"y: {y.shape}")'''


def workflow(model, epoch, device, sv_data=None):

    layer_preparation(model)
    epochs(model, epoch, device, sv_data)
    freezing(model)


def freezing(model):
    for i in model.list_of_layers[-1][0].parameters():
        i.requires_grad = False
    model.list_of_layers[-1][0].eval()


def layer_preparation(model):
    model.list_of_layers[-1][0].train()
    for i in model.list_of_layers[-1][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True


def epochs(model, epoch, device, sv_data=None):
    # todo: only for first Chapter of Classification
    train_dataset, valid_dataset = mnist_data_loader(True, 128)
    optimizer = torch.optim.SGD(model.list_of_layers[-1][0].parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    # crit = torch.nn.NLLLoss()
    i = 0
    while i < epoch:
        train_epoch(train_dataset, model, device, crit, optimizer, sv_data)
        test_new_layer(valid_dataset, model, device, crit, sv_data)
        i += 1


def train_epoch(dataset, model, device, crit, optimizer, sv_data=None):
    model.list_of_layers[-1][0].to(device)

    for data, label in dataset:
        optimizer.zero_grad()
        # Data through the whole Network
        output = model(data.to(device))

        loss = crit(output, label)
        loss.backward()
        optimizer.step()


def test_new_layer(test_dataset, model, device, loss_fn, sv_data=None):
    # todo: This function is from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    size = len(test_dataset.dataset)
    num_batches = len(test_dataset)
    # From here the half is from me
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in test_dataset:
            output = model(data.to(device))
            # todo: The rest of the function is from pytorch.org
            test_loss += loss_fn(output, label).item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def mnist_data_loader(train, batch_size):
    # todo: For TF we need another data_loader
    dataset = torchvision.datasets.MNIST(root='', train=train, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000, 10000])
    data_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_load_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)
    return data_load, data_load_valid


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
