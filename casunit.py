# Here are the Units for adding in the Network

import torch
import numpy


class CascadeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.list_of_layers = []

    def forward(self, x):
        for layer in self.list_of_layers:
            if layer[1] == 'relu':
                x = torch.nn.functional.relu(layer[0](x))
            elif layer[1] == 'reshape':
                x = x.reshape(layer[0])
            else:
                x = layer[0](x)
        return x

    def add_layer(self, neural_node, activation=None):
        seclis = [neural_node, activation]
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
