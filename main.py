# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 16.04.2025
# Version: 0.0.002

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

    epoch = 1
    # todo: hier eine Iteration in der alle Layer bzgl. des gerade Gegenwärtigen Losses hinzufügen.
    # todo: Problem: es dauert immer länger hier, je länger das Netz wird.
    model.add_layer(torch.nn.Linear(28, 32).to(device))
    workflow(model, epoch, device)
    # sv_data = []
    # sv_data = save_output(model, sv_data, True)

    model.add_layer(torch.nn.Conv2d(1, 10, 5).to(device), 'relu')
    workflow(model, epoch, device)
    # sv_data = save_output(model, sv_data)

    model.add_layer(torch.nn.Conv2d(10, 10, 5).to(device), 'relu')
    workflow(model, epoch, device)
    # sv_data = save_output(model, sv_data)

    # todo: Transfer-learning Wechsel zu dem nächsten Netzwerk, welches ein Cascade ist.

    '''train_dataset = data_loader(True, 64)
    test_dataset = data_loader(False, 64)

    for x, y in test_dataset:
        print(f"x: {x.shape}")  # 1, 1, 28, 28
        print(f"y: {y.shape}")'''


def save_output(model, list_data, first=None):
    # todo: das hier muss irgendwie zu einem Tensor verwandelt werden.
    if first:
        dataset = mnist_data_loader(False, 1)
        for data in dataset:
            x = torch.FloatTensor(1, 1, 28, 28)
            x = model.list_of_layers[-1][0](data)
            list_data.append(x)
            # list_data.append(model.list_of_layers[-1][0](data))
    else:
        i = 0
        for data in list_data:
            list_data[i] = model.list_of_layers[-1][0](data)
            i += 1
    return list_data


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
    train_dataset = mnist_data_loader(True, 1)
    test_dataset = mnist_data_loader(False, 1)
    optimizer = torch.optim.SGD(model.list_of_layers[-1][0].parameters(), lr=0.01)
    crit = torch.nn.HingeEmbeddingLoss()
    i = 0
    while i < epoch:
        train_epoch(train_dataset, model, device, crit, optimizer, sv_data)
        test_new_layer(test_dataset, model, device, crit, sv_data)
        i += 1


def train_epoch(dataset, model, device, crit, optimizer, sv_data=None):
    model.list_of_layers[-1][0].to(device)
    for data, label in dataset:
        optimizer.zero_grad()
        output = data.to(device)

        for layer in model.list_of_layers:
            # todo: ist es sinnvoll, den Kram irgendwo zwischenzuspeichern? -> Ja, ist es
            # todo: Moglichkeit: eine Liste, shuffle wäre dann ein Problem.
            output = layer[0](output)

        loss = crit(output, label.float())
        # todo: Maybe i need something else than backprop
        optimizer.step()
        loss.backward()


def test_new_layer(test_dataset, model, device, loss_fn, sv_data=None):
    # todo: This function is from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    size = len(test_dataset.dataset)
    num_batches = len(test_dataset)
    # From here the half is from me
    model.list_of_layers[-1][0].eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in test_dataset:
            data, label = data.to(device), label.to(device)
            output = data
            for layer in model.list_of_layers:
                output = layer[0](output)
            # todo: The rest of the function is from pytorch.org
            test_loss += loss_fn(output, label.float()).item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def mnist_data_loader(train, batch_size):
    # todo: For TF we need another data_loader
    dataset = torchvision.datasets.MNIST(root='', train=train, download=True, transform=torchvision.transforms.ToTensor())
    data_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_load


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
