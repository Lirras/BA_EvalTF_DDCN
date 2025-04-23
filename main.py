# Tips
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

# Title: Bachelor-Arbeit: Evaluierung von Transferlernen mit Deep Direct Cascade Networks
# Author: Simon Tarras
# Date: 23.04.2025
# Version: 0.0.004

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
    # Length of Time is Acceptable

    # Load the MNIST Dataset
    test_data, valid_data = mnist_data_loader(True, 100)

    # Start of Cascade Network
    model.add_layer(torch.nn.Identity())  # Identity is needed for changes at the Input shapes, due to the different
    # Datasets
    model.add_layer((-1, 784), 'reshape')
    model.add_layer(torch.nn.Linear(784, 784, device=device))
    workflow(model, epoch, device, test_data, valid_data)

    # todo: With TF is here an error
    # todo: The Datasets are 40000, 10000, 73257, 26032 -> Batch_size=100 -> Size change at last iteration of Epoch
    model.add_layer((100, 1, 28, 28), 'reshape')  # Size 44688

    # todo: Why is Conv so much worse than Linear?
    model.add_layer(torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, padding_mode='zeros', device=device), 'relu')
    # model.add_layer(1, 'max_pooling')
    model.add_layer((-1, 784), 'reshape')
    workflow(model, epoch, device, test_data, valid_data)

    model.add_layer(torch.nn.Linear(784, 10, device=device))  # A Linear Layer has only Sense as One Layer and
    # with few epochs. Kinda 1-5
    workflow(model, epoch, device, test_data, valid_data)

    # todo: make the correct Shape of outputs at here with reshaping for output: 1, size with label: 1-Dim

    # Begin of TF:
    new_test_data, new_valid_data = svhn_data_loader(100)  # Maybe Cut the Last Data?

    # for x, y in new_test_data:  # 128, 1, 28, 28 <- Vorher 1 Channel, weil Schwarz-Weiß
    #     print(f"x: {x.shape}")  # 128, 3, 32, 32/ B, C, H, W 3 Channel, weil RGB-Farben
    # todo Es braucht ein Conv(1, 3, 5), damit aus 32x32 ein 28x28 wird. Dies vor dem Reshape -> Batch_size
    #  Möglichkeit überprüfen

    # This is the Downscaling for in, out, kernel
    model.list_of_layers[0] = [torch.nn.Conv2d(3, 1, 5, stride=1, padding=0, padding_mode='zeros',
                                               device=device), 'relu']
    for i in model.list_of_layers[0][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True
    # model.list_of_layers[0] = [(-1, (784, 10)), 'reshape']  # todo: Downscaling
    # todo: Wenn ich dafür mehr als 1 Layer brauche, kann ich Identities vorher hinzufügen. -> Kostet Performance
    # model.add_layer(torch.nn.Linear(10, 10).to(device))
    workflow(model, epoch, device, new_test_data, new_valid_data)
    # todo: This causes Error at first layer. (at Reshape) -> I need b, c, h, w with 100 1 28 28 not 100 3 32 32

    # sv_data = save_output(model, sv_data)

    # todo: Transfer-learning Wechsel zu dem nächsten Netzwerk, welches ein Cascade ist.


def workflow(model, epoch, device, train, valid):

    layer_preparation(model)
    epochs(model, epoch, device, train, valid)
    freezing(model)


def freezing(model):
    # This is functioning

    last_layer = get_last_layer_params(model.list_of_layers)
    for i in model.list_of_layers[last_layer][0].parameters():
        i.requires_grad = False
    model.list_of_layers[last_layer][0].eval()


def layer_preparation(model):

    model.train()
    for i in model.list_of_layers[get_last_layer_params(model.list_of_layers)][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True


def get_last_layer_params(list_of_layers):
    # Backward Iteration for Seeking the Last Layer with Weight Parameters
    x = len(list_of_layers)-1
    while x >= 0:
        if list_of_layers[x][1] == 'relu' or list_of_layers[x][1] is None:
            return x
        else:
            x -= 1
    print("Error at get last Layer Params!")
    return -2  # Here is an Error


def epochs(model, epoch, device, train_dataset, valid_dataset):

    optimizer = torch.optim.SGD(model.list_of_layers[get_last_layer_params(
        model.list_of_layers)][0].parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    # There is a case, that this is logSoftmax followed by NLLLoss

    # crit = torch.nn.HingeEmbeddingLoss() check how identical the tensors are; That is Senseless
    # crit = torch.nn.NLLLoss()  # Also Senseless, cause CrossEntropyLoss makes this in it too
    i = 0
    while i < epoch:
        train_epoch(train_dataset, model, device, crit, optimizer)
        test_new_layer(valid_dataset, model, device, crit)
        i += 1


def train_epoch(dataset, model, device, crit, optimizer):
    model.to(device)
    counter = 1
    print(len(dataset))
    for data, label in dataset:
        # todo: this does exactly what i want, but it isnt enough!
        if counter == len(dataset):
            print(data.shape)
            break
        optimizer.zero_grad()
        # Data through the whole Network
        output = model(data.to(device))
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        counter += 1
    print(counter)


def test_new_layer(test_dataset, model, device, loss_fn):
    # This function is from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    # todo: size and num_batches are wrong now -> Await little less ACC
    size = len(test_dataset.dataset)
    num_batches = len(test_dataset)
    # From here the half is from me
    model.eval()
    counter = 0
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in test_dataset:
            if counter == len(test_dataset):
                break
            output = model(data.to(device))
            # The rest of the function is from pytorch.org
            test_loss += loss_fn(output, label).item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
            counter += 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def mnist_data_loader(train, batch_size):
    # 1st Dataloader for Source Dataset at Classification Domain TF
    dataset = torchvision.datasets.MNIST(root='/MNIST', train=train, download=True,
                                         transform=torchvision.transforms.ToTensor())
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000, 10000])
    data_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_load_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return data_load, data_load_valid


def svhn_data_loader(batch_size):
    # 2nd Dataloader for Target Dataset at Classification Domain TF
    train_dataset = torchvision.datasets.SVHN(root='/SVHN', split='train', download=True,
                                              transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.SVHN(root='/SVHN', split='test', download=True,
                                             transform=torchvision.transforms.ToTensor())
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_load, test_load


def length_of_data(length, batch_size):
    x = len(length) / batch_size
    # y = x
    # delete_amount = (y - int(x)) * batch_size
    return int(x)


# This is from Fully Convolutional Neural Network Structure
# and Its Loss Function for Image Classification by QIUYU ZHU , (Member, IEEE), AND XUEWEN ZU
# This is disfunctional:
def softmax_loss(batch_size, output, label):
    return (1/batch_size)*sum(-math.log(math.exp(label)/sum(math.exp(output))))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
