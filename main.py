# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

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

    epoch = 5
    # todo: hier eine Iteration in der alle Layer bzgl. des gerade Gegenwärtigen Losses hinzufügen.
    # todo: Problem: es dauert immer länger hier, je länger das Netz wird.

    test_data, valid_data = mnist_data_loader(True, 128)
    # This Change changes nothing, but its more variable

    # tensor = torch.FloatTensor(torch.randn((1, 2, 3)))
    # print(tensor.shape)
    # tensor = torch.unsqueeze(tensor, 0)
    # print(tensor.shape)

    # todo: Why is this after Conv always the Same Result? -> It was only one input, and is only one output
    # todo: But its a minibatch! There must be 128 different ones!
    # todo: Batch_size=1 funktioniert besser, aber warum?
    model.add_layer(torch.nn.Conv2d(1, 1, 5, stride=1, padding=2, padding_mode='zeros', device=device), 'relu')
    model.add_layer(2, 'max_pooling')
    workflow(model, epoch, device, test_data, valid_data)

    # model.add_layer((-1, 784), 'reshape')
    # model.add_layer(1, 'unsqueeze')
    # model.add_layer(1, 'unsqueeze')
    # model.add_layer(1, 'squeeze')
    # model.add_layer((2, 2), 'max_pooling')
    # model.add_layer(torch.nn.Conv2d(1, 1, 5), 'relu')
    # workflow(model, epoch, device, test_data, valid_data)

    # todo: Zuerst Conv alles und dann zum Schluss ein reshape machen für den loss.
    # todo: der muss dann aber in jedem Layer verändert werden. -> Da muss ja einiges verändert werden...

    # input_size = 28*28
    # model.add_layer((-1, 784), 'reshape')
    # model.add_layer(torch.nn.Linear(input_size, 10).to(device))
    # workflow(model, epoch, device, test_data, valid_data)

    new_test_data, new_valid_data = svhn_data_loader(128)
    # for x, y in new_test_data:  # 128, 1, 28, 28 <- Vorher 1 Channel, weil Schwarz-Weiß
    #     print(f"x: {x.shape}")  # 128, 3, 32, 32/ B, C, H, W 3 Channel, weil RGB-Farben
        # todo Es braucht ein Conv(1, 3, 5), damit aus 32x32 ein 28x28 wird. Dies vor dem Reshape -> Batch_size
        #  Möglichkeit überprüfen
    #     break

    # model.list_of_layers[0] = [(-1, (784, 10)), 'reshape']  # todo: Downscaling
    # todo: Wenn ich dafür mehr als 1 Layer brauche, kann ich Identities vorher hinzufügen. -> Kostet Performance
    # model.add_layer(torch.nn.Linear(10, 10).to(device))
    # workflow(model, epoch, device, new_test_data, new_valid_data)

    # model.add_layer(torch.nn.Conv2d(1, 1, 5).to(device))  # todo: For RGB: 3, 1, 5 / In, Out, Kernel
    # workflow(model, epoch, device, test_data, valid_data)

    # sv_data = save_output(model, sv_data)

    # todo: Transfer-learning Wechsel zu dem nächsten Netzwerk, welches ein Cascade ist.

    '''train_dataset = data_loader(True, 64)
    test_dataset = data_loader(False, 64)

    for x, y in test_dataset:
        print(f"x: {x.shape}")  # 1, 1, 28, 28
        print(f"y: {y.shape}")'''


def workflow(model, epoch, device, train, valid, sv_data=None):

    layer_preparation(model)
    epochs(model, epoch, device, train, valid, sv_data)
    freezing(model)


def freezing(model):
    # This is functioning
    if model.list_of_layers[-1][1] == 'max_pooling':
        for i in model.list_of_layers[len(model.list_of_layers)-2][0].parameters():
            i.requires_grad = False
        model.list_of_layers[len(model.list_of_layers)-2][0].eval()
    else:
        for i in model.list_of_layers[-1][0].parameters():
            i.requires_grad = False
        model.list_of_layers[-1][0].eval()


def layer_preparation(model):
    # model.list_of_layers[-1][0].train()
    model.train()
    if model.list_of_layers[-1][1] == 'max_pooling':
        for i in model.list_of_layers[len(model.list_of_layers)-2][0].parameters():
            if hasattr(i, 'grad_fn'):
                i.requires_grad = True
    else:
        for i in model.list_of_layers[-1][0].parameters():
            if hasattr(i, 'grad_fn'):
                i.requires_grad = True


def epochs(model, epoch, device, train_dataset, valid_dataset, sv_data=None):
    # todo: only for first Chapter of Classification
    # train_dataset, valid_dataset = mnist_data_loader(True, 128)
    if model.list_of_layers[-1][1] == 'max_pooling':
        optimizer = torch.optim.SGD(model.list_of_layers[len(model.list_of_layers)-2][0].parameters(), lr=0.01)
    else:
        optimizer = torch.optim.SGD(model.list_of_layers[-1][0].parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()  # Das ist der logisch Sinnvollste Loss
    # crit = torch.nn.HingeEmbeddingLoss() Der vergleicht, ob die Tensoren gleich sind; Ist also sinnlos.
    # crit = torch.nn.NLLLoss()
    # crit = None
    i = 0
    while i < epoch:
        train_epoch(train_dataset, model, device, crit, optimizer, sv_data)
        test_new_layer(valid_dataset, model, device, crit, sv_data)
        i += 1


def train_epoch(dataset, model, device, crit, optimizer, sv_data=None):
    # model.list_of_layers[-1][0].to(device)
    model.to(device)

    for data, label in dataset:
        optimizer.zero_grad()
        # Data through the whole Network
        output = model(data.to(device))
        # print(label.shape)
        # print(output.shape)
        # print(output)
        # todo: make the correct Shape of outputs at here with reshaping for output: 1, size with label: 1-Dim
        output = output.reshape(-1, 196)  # 1, 144
        # print(output.shape)
        # print(label.shape)
        loss = crit(output, label)
        # loss = softmax_loss(1, output, label)
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
            output = output.reshape(-1, 196)
            # todo: The rest of the function is from pytorch.org
            test_loss += loss_fn(output, label).item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    print(test_loss, num_batches)
    correct /= size
    print(correct, size)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def mnist_data_loader(train, batch_size):
    # 1st Dataloader for Source Dataset at Classification Domain TF
    dataset = torchvision.datasets.MNIST(root='/MNIST', train=train, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000, 10000])
    data_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_load_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return data_load, data_load_valid


def svhn_data_loader(batch_size):
    # 2nd Dataloader for Target Dataset at Classification Domain TF
    train_dataset = torchvision.datasets.SVHN(root='/SVHN', split='train', download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.SVHN(root='/SVHN', split='test', download=True, transform=torchvision.transforms.ToTensor())
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_load, test_load


# This is from Fully Convolutional Neural Network Structure
# and Its Loss Function for Image Classification by QIUYU ZHU , (Member, IEEE), AND XUEWEN ZU
# This is disfunctional:
def softmax_loss(batch_size, output, label):
    return (1/batch_size)*sum(-math.log(math.exp(label)/sum(math.exp(output))))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
