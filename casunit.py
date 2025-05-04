# Here are the Units for adding in the Network

import torch
import numpy


class CascadeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.list_of_layers = []

    def forward(self, x):
        counter = 1
        for layer in self.list_of_layers:
            # print(counter)
            # counter += 1
            # todo: Das geht nur mit Dingen, die keine Parameter haben
            # todo: Baue das als Switch-Case um
            match layer[1]:
                case 'relu':
                    x = torch.nn.functional.relu(layer[0](x))
                case 'reshape':
                    x = x.reshape(layer[0])
                case 'squeeze':
                    x = x.squeeze(layer[0])
                case 'unsqueeze':
                    x = x.unsqueeze(layer[0])
                case 'max_pooling':
                    x = torch.nn.functional.max_pool2d(x, kernel_size=layer[0])
                case 'flatten':
                    x = torch.flatten(x, layer[0])
                case 'unflatten':
                    x = torch.unflatten(x, layer[0], layer[2])
                case 'batch_norm':
                    y = layer[2][0].to(layer[3])
                    z = layer[2][1].to(layer[3])
                    x = torch.nn.functional.batch_norm(x, y, z)
                case _:
                    x = layer[0](x)
        return x

    def add_layer(self, neural_node, activation=None, secondary=None, dev=None):
        seclis = [neural_node, activation, secondary, dev]
        self.list_of_layers.append(seclis)


# todo: wird nicht genutzt, da es gerade nicht funktioniert
class NodeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        # self.ident = torch.nn.Identity(20)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        # x = self.ident(x)
        return x


class svhn_network(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.nn.Linear(1024, 1024)(x)
        x = torch.unflatten(x, 1, (32, 32))
        x = torch.unsqueeze(x, 1)


        return x
