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
    # todo: Wieso ist das Ergebnis danach immer größer 1?
    input_size = 28*28
    model.add_layer((-1, 784), 'reshape')
    model.add_layer(torch.nn.Linear(input_size, 10).to(device))
    workflow(model, epoch, device)
    # sv_data = []
    # sv_data = save_output(model, sv_data, True)
    # model.add_layer(torch.nn.Linear(128, 10).to(device))
    # workflow(model, epoch, device)
    # model.add_layer(torch.nn.Conv2d(1, 1, 5).to(device), 'relu')
    # workflow(model, epoch, device)
    # sv_data = save_output(model, sv_data)

    # model.add_layer(torch.nn.Conv2d(10, 10, 5).to(device), 'relu')
    # workflow(model, epoch, device)
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
    train_dataset, valid_dataset = mnist_data_loader(True, 128)
    optimizer = torch.optim.SGD(model.list_of_layers[-1][0].parameters(), lr=0.01)
    crit = torch.nn.NLLLoss()
    i = 0
    while i < epoch:
        train_epoch(train_dataset, model, device, crit, optimizer, sv_data)
        test_new_layer(valid_dataset, model, device, crit, sv_data)
        i += 1


def train_epoch(dataset, model, device, crit, optimizer, sv_data=None):
    model.list_of_layers[-1][0].to(device)
    # print(model.list_of_layers[-1][0].weight)  # 1, 1, 5, 5
    for data, label in dataset:
        # data.squeeze()
        # print(data.shape)
        # print(label.shape)
        optimizer.zero_grad()
        output = model(data.to(device))
        # probability_row = torch.nn.Softmax(output)
        # print(probability_row)
        # print(torch.argmax(output, 1))
        # print(accuracy(output, label))

        # for layer in model.list_of_layers:
            # todo: ist es sinnvoll, den Kram irgendwo zwischenzuspeichern? -> Ja, ist es
            # todo: Moglichkeit: eine Liste, shuffle wäre dann ein Problem.
            # output = layer[0](output)

        # print(output.shape)
        # print(label.shape)

        # weight_tensor = model.list_of_layers[-1][0].weight
        # print(weight_tensor)
        # todo: The target need 3D.
        # loss = torch.nn.functional.nll_loss(output, label, weight_tensor)
        loss = crit(output, label)
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
            output = model(data.to(device))
            '''data, label = data.to(device), label.to(device)
            output = data
            for layer in model.list_of_layers:
                output = layer[0](output)'''
            # todo: The rest of the function is from pytorch.org
            test_loss += loss_fn(output, label).item()
            # print((output.argmax(1) == label).type(torch.float).sum().item())
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# todo: Function is from https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return (torch.tensor(torch.sum(preds == labels).item()/ len(preds)))


def mnist_data_loader(train, batch_size):
    # todo: For TF we need another data_loader
    dataset = torchvision.datasets.MNIST(root='', train=train, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000, 10000])
    data_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_load_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)
    # image_tensor, label = dataset[0]
    # print(image_tensor.shape, label)
    return data_load, data_load_valid


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
