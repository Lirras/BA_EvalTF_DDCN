import train_and_test as tnt
import torch


def workflow(model, epoch, device, train, valid):
    print("__________________________________Next Layer___________________________________")
    layer_preparation(model)
    epochs(model, epoch, device, train, valid)
    freezing(model)


def freezing(model):
    # This is functioning
    # todo: Does have Reshape, Pooling etc. Params?
    last_layer = get_last_layer_params(model.list_of_layers)
    for i in model.list_of_layers[last_layer][0].parameters():
        i.requires_grad = False
    # model.list_of_layers[last_layer][0].eval()
    model.eval()


def layer_preparation(model):

    model.train()  # todo: This is not, what i want
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
        tnt.train_epoch(train_dataset, model, device, crit, optimizer)
        tnt.test_new_layer(valid_dataset, model, device, crit)
        i += 1
