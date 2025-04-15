# This is a sample Python script.
import netofnets
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from casunit import CascadeNetwork
import torch
import torchvision
import numpy
from netofnets import NetOfNets
from casunit import NodeNetwork


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Strg+F8 to toggle the breakpoint.

    model = CascadeNetwork()

    # dataset = data_loader()

    # todo: define next layer after training of the last layer by adding new calls to this method
    new_train_of_layer(True, model, torch.nn.Conv2d(1, 20, 5), 'relu')
    new_train_of_layer(True, model, torch.nn.Conv2d(1, 20, 5), 'relu')
    # new_train_of_layer(model, torch.nn.AvgPool1d(3, 2))
    # todo: return loss; Abbruch, wenn klein genug, dann das nächste Netz machen mit einem anderen Dataloader und
    # todo: vorhergehendem ersten Netz. Spalten der Datensätze anpassen.


def new_train_of_layer(first, model, layer, activation=None):
    model.add_layer(layer, activation)

    dataset = data_loader()

    # For new Layers; enable Training update
    for i in model.list_of_layers[-1][0].parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True

    model.list_of_layers[-1][0].train()

    optimizer = torch.optim.SGD(model.list_of_layers[-1][0].parameters(), lr=0.01)
    # In training loop
    # this is a Epoch for all layers of the Network.
    # todo: epoch einstellen, evtl. dynamisch
    loss = 1
    epoch = 0
    counter = 0
    # output = None
    while counter <= epoch:
        print(counter)
        # todo: dataset hat nur im ersten Layer ein label
        for data, label in dataset:
            optimizer.zero_grad()
            output = None
            # todo: Das durchgeben aller Daten für die späteren Iterationen muss besser gehen, weil das ewig dauert!
            for layer in model.list_of_layers:
                if first:
                    # Only for the first Layer:
                    output = layer[0](data)
                else:
                    output = layer[0](output)
            # output = model.list_of_layers[-1][0](data)
            # crit = torch.nn.CrossEntropyLoss()
            crit = torch.nn.HingeEmbeddingLoss()
            loss = crit(output, label.float())
            optimizer.step()
            # Nur das letzte Layer wird geupdatet, da die anderen nicht betrachtet werden, obwohl das Backprop ist.
            loss.backward()
        counter += 1
    print(loss)

    model.list_of_layers[-1][0].eval()

    # todo: Freezing überprüfen!!!

    # todo: This is freezing und vergleiche mit dem Neuen Gewichten.
    # todo: Braucht noch Weights vor diesen Zeilen!
    # freeze the last Layer, todo: only after training the Layer
    # Freezing for all Params:
    for i in model.list_of_layers[-1][0].parameters():
        i.requires_grad = False

    # return output


def test_net_of_Nets_as_Cascade_Approach():
    # todo: Das hier funktioniert nicht, weil grad_fn fehlt. Dürfte daran liegen, dass es einzelne Netze sind, die nur
    # todo: ein Layer haben.
    network = NetOfNets()
    layer1 = NodeNetwork()

    dataset = data_loader()

    optimizer = torch.optim.SGD(layer1.parameters(), lr=0.01)

    layer1.train()

    for i in layer1.parameters():
        if hasattr(i, 'grad_fn'):
            i.requires_grad = True

    # training:
    loss = 1
    while loss > 0.1:
        for data, label in dataset:
            print("test")
            # zero_grad wegen Gradientenübertrag von vorheriger Epoche
            optimizer.zero_grad()
            # prediction
            # todo: überarbeiten, weil das könnte dauern; vllt. outputs zwischenspeichern?
            output = data
            for layer in network.list_of_nets:
                output = layer(output)

            # output = layer1(data)
            # loss criterion
            crit = torch.nn.MSELoss()
            # real loss
            loss = crit(output, label.float())
            # Übertragen und update
            # loss.backward()
            optimizer.step()
            loss.backward()
            break
        # todo: delete this line
        loss = 0

    # todo: That is the last step, after whole training
    network.add_network(layer1)
    # todo: data through all layers before. Need to freeze weights. This could be freeze
    for layer in network.list_of_nets:
        layer.eval()


def data_loader():
    # todo: Für Transferlearning einen anderen schreiben
    # todo: Größe der Batch-Size verändern
    dataset = torchvision.datasets.MNIST(root='', train=True, download=True, transform=torchvision.transforms.ToTensor())
    data_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return data_load


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_net_of_Nets_as_Cascade_Approach()
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
