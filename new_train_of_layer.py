from main import mnist_data_loader
from main import test_new_layer
import torch


def new_train_of_layer(device, model, layer, activation=None):
    model.add_layer(layer, activation)

    dataset = mnist_data_loader(True, 1)
    test_dataset = mnist_data_loader(False, 1)

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
        for data, label in dataset:
            # data.to('cuda:0')
            label.to(device)
            optimizer.zero_grad()
            output = data
            # print(data.is_cuda)
            # print(output.is_cuda)
            # todo: Das durchgeben aller Daten für die späteren Iterationen muss besser gehen, weil das ewig dauert!
            for layer in model.list_of_layers:
                # layer[0].to('cuda')
                output = layer[0](output)
            # output = model.list_of_layers[-1][0](data)
            # crit = torch.nn.CrossEntropyLoss()
            crit = torch.nn.HingeEmbeddingLoss()
            loss = crit(output, label.float())
            optimizer.step()
            # Nur das letzte Layer wird geupdatet, da die anderen nicht betrachtet werden, obwohl das Backprop ist.
            loss.backward()
        test_new_layer(test_dataset, model, device, torch.nn.HingeEmbeddingLoss())
        counter += 1
    # print(loss)

    model.list_of_layers[-1][0].eval()

    # todo: Freezing überprüfen!!!

    # todo: This is freezing und vergleiche mit dem Neuen Gewichten.
    # todo: Braucht noch Weights vor diesen Zeilen!
    # freeze the last Layer, todo: only after training the Layer
    # Freezing for all Params:
    for i in model.list_of_layers[-1][0].parameters():
        i.requires_grad = False

    # return output
