from netofnets import NetOfNets
from casunit import NodeNetwork
from main import mnist_data_loader
import torch

def test_net_of_Nets_as_Cascade_Approach():
    # todo: Das hier funktioniert nicht, weil grad_fn fehlt. Dürfte daran liegen, dass es einzelne Netze sind, die nur
    # todo: ein Layer haben.
    network = NetOfNets()
    layer1 = NodeNetwork()

    dataset = mnist_data_loader(True, 1)

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

# This is from Fully Convolutional Neural Network Structure
# and Its Loss Function for Image Classification by QIUYU ZHU , (Member, IEEE), AND XUEWEN ZU
# This is disfunctional:
def softmax_loss(batch_size, output, label):
    return (1/batch_size)*sum(-math.log(math.exp(label)/sum(math.exp(output))))
